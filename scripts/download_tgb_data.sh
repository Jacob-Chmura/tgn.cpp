#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

if ! command -v uv >/dev/null 2>&1; then
    echo "uv is not installed. Install it first: https://docs.astral.sh/uv/getting-started/installation/"
    exit 1
fi

DATASET_NAME="$1"
DEST_DATA_DIR="$PROJECT_ROOT/data"
mkdir -p "$DEST_DATA_DIR"

uv run --no-project --with py-tgb --with numpy python - <<EOF
import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR

name, dest = "$DATASET_NAME", "$DEST_DATA_DIR"

print(f"Downloading {name}...")
ds = (NodePropPredDataset if name.startswith('tgbn-') else LinkPropPredDataset)(name=name)
data, masks = ds.full_data, {'train': ds.train_mask, 'val': ds.val_mask, 'test': ds.test_mask}
src, dst, ts = data['sources'], data['destinations'], data['timestamps']

cols, header, fmts = [src, dst, ts], "src,dst,time", ['%d']*3

if data['edge_feat'] is not None:
    cols.append(data['edge_feat'])
    header += "," + ",".join(f"msg_{i}" for i in range(data['edge_feat'].shape[1]))
    fmts += ['%g'] * data['edge_feat'].shape[1]

if name.startswith('tgbl-'):
    max_neg = n_neg = 1000
    ns = NegativeEdgeSampler(dataset_name=name)
    v = f'_v{DATA_VERSION_DICT[name]}' if DATA_VERSION_DICT.get(name, 1) > 1 else ''
    full_negs = np.full((len(src), max_neg), -1, dtype=np.int32)

    for split in ['val', 'test']:
        print(f"Processing negatives for {split}...")
        ns_path = f"{PROJ_DIR}datasets/{name.replace('-', '_')}/{name}_{split}_ns{v}.pkl"
        ns.load_eval_set(fname=ns_path, split_mode=split)

        negs = ns.query_batch(src[masks[split]], dst[masks[split]], ts[masks[split]], split_mode=split)
        cur_n_neg = min(len(x) for x in negs)
        if cur_n_neg > max_neg:
            print(f"Found {cur_n_neg} negatives per positive in {split} split, truncating to {max_neg}")

        n_neg = min(n_neg, cur_n_neg)
        full_negs[masks[split], :n_neg] = [x[:n_neg] for x in negs]

    cols.append(full_negs[:, :n_neg])
    header += "," + ",".join(f"neg_{i}" for i in range(n_neg))
    fmts += ['%d'] * n_neg

out_path = f"{dest}/{name}.csv"
print(f"Saving {name} ({len(src)} edges) to {out_path}...")
np.savetxt(out_path, np.column_stack(cols), delimiter=",", header=header, comments='', fmt=fmts)
print("Done.")
EOF
