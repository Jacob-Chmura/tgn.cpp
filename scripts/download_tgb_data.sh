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

name = "$DATASET_NAME"
out_path = f"$DEST_DATA_DIR/{name}.csv"
print(f"Downloading TGB dataset: '{name}'...")

is_link = name.startswith('tgbl-')
ds = NodePropPredDataset(name=name) if name.startswith('tgbn-') else LinkPropPredDataset(name=name)
data = ds.full_data

src, dst, time = data['sources'], data['destinations'], data['timestamps']
cols = [src, dst, time]
header = "src,dst,time"
formats = ['%d', '%d', '%d']

if data['edge_feat'] is not None:
    msg_dim = data['edge_feat'].shape[1]
    cols.append(data['edge_feat'])
    header += "," + ",".join(f"msg_{i}" for i in range(msg_dim))
    formats += ['%g'] * msg_dim

if is_link:
    ns = NegativeEdgeSampler(dataset_name=name)
    v_suffix = f'_v{DATA_VERSION_DICT[name]}' if DATA_VERSION_DICT.get(name, 1) > 1 else ''

    queries = {}
    n_neg = float('inf')
    splits = {'val': ds.val_mask, 'test': ds.test_mask}

    for split, mask in splits.items():
        print(f"Loading and querying {split} negatives...")
        ns_path = f"{PROJ_DIR}datasets/{name.replace('-', '_')}/{name}_{split}_ns{v_suffix}.pkl"
        ns.load_eval_set(fname=ns_path, split_mode=split)

        queries[split] = ns.query_batch(src[mask], dst[mask], time[mask], split_mode=split)
        n_neg = min(n_neg, *(len(x) for x in queries[split]))

    print(f"Global minimum negatives found: {n_neg}.")
    negs = np.full((len(src), n_neg), -1, dtype=np.int32)
    for split, mask in splits.items():
        negs[mask] = np.array([x[:n_neg] for x in queries.pop(split)], dtype=np.int32)

    cols.append(negs)
    header += "," + ",".join(f"neg_{i}" for i in range(n_neg))
    formats += ['%d'] * n_neg

print(f"Saving data ({len(src)} edges) to '{out_path}'...")
np.savetxt(out_path, np.column_stack(cols), delimiter=",", header=header, comments='', fmt=formats)
print("Done.")
EOF
