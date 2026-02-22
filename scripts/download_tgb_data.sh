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
DEST_DATA_DIR="$PROJECT_ROOT/data/$DATASET_NAME"

mkdir -p "$DEST_DATA_DIR"

uv run --no-project --with py-tgb --with numpy python - <<EOF
import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.linkproppred.negative_sampler import NegativeEdgeSampler
from tgb.nodeproppred.dataset import NodePropPredDataset
from tgb.utils.info import DATA_VERSION_DICT, PROJ_DIR

name, dest = "$DATASET_NAME", "$DEST_DATA_DIR"
print(f"Loading TGB dataset: {name}")

is_link = name.startswith('tgbl-')
ds = NodePropPredDataset(name=name) if name.startswith('tgbn-') else LinkPropPredDataset(name=name)
data = ds.full_data

masks = {'train': ds.train_mask, 'val': ds.val_mask, 'test': ds.test_mask}
ns = NegativeEdgeSampler(dataset_name=name) if is_link else None

for split, mask in masks.items():
    s_src, s_dst, s_time = data['sources'][mask], data['destinations'][mask], data['timestamps'][mask]

    cols = [s_src, s_dst, s_time]
    header = "src,dst,time"
    formats = ['%d', '%d', '%d']

    if data['edge_feat'] is not None:
        cols.append(data['edge_feat'][mask])
        msg_dim = data['edge_feat'].shape[1]
        header += "," + ",".join(f"msg_{i}" for i in range(msg_dim))
        formats += ['%g'] * msg_dim

    if ns and split in ['val', 'test']:
        print(f"Extracting negatives for {split}")

        v_suffix = f'_v{DATA_VERSION_DICT[name]}' if DATA_VERSION_DICT.get(name, 1) > 1 else ''
        ns_path = f"{PROJ_DIR}datasets/{name.replace('-', '_')}/{name}_{split}_ns{v_suffix}.pkl"
        ns.load_eval_set(fname=ns_path, split_mode=split)

        # TODO(kuba) We only work with homogenous number of negatives per positive, for now
        negs = ns.query_batch(s_src, s_dst, s_time, split_mode=split)
        n_neg = min(len(x) for x in negs)
        neg_array = np.array([x[:n_neg] for x in negs], dtype=np.int32)

        cols.append(neg_array)
        header += "," + ",".join(f"neg_{i}" for i in range(n_neg))
        formats += ['%d'] * n_neg

    out_path = f"{dest}/{split}.csv"
    np.savetxt(out_path, np.column_stack(cols), delimiter=",", header=header, comments='', fmt=formats)
    print(f"Saved {split} split ({len(s_src)} edges) to {out_path}")

print("Done.")
EOF
