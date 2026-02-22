#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

DATASET_NAME="$1"
DEST_DATA_DIR="$PROJECT_ROOT/data/$DATASET_NAME"

mkdir -p "$DEST_DATA_DIR"

uv run --no-project --with py-tgb --with numpy python - <<EOF
import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset

name, dest = "$DATASET_NAME", "$DEST_DATA_DIR"
print(f"Loading TGB dataset: {name}")

ds = NodePropPredDataset(name=name) if name.startswith('tgbn-') else LinkPropPredDataset(name=name)
data = ds.full_data

masks = {
    'train': ds.train_mask,
    'val': ds.val_mask,
    'test': ds.test_mask
}

for split, mask in masks.items():
    data_cols = [data['sources'][mask], data['destinations'][mask], data['timestamps'][mask]]
    header = "src,dst,time"

    if data['edge_feat'] is not None:
        data_cols.append(data['edge_feat'][mask])
        header += "," + ",".join(f"msg_{i}" for i in range(data['edge_feat'].shape[1]))

    out_path = f"{dest}/{split}.csv"
    np.savetxt(out_path, np.column_stack(data_cols), delimiter=",", header=header, comments='', fmt='%g')
    print(f"  - Saved {split} split ({len(data_cols[0])} edges) to {out_path}")
EOF
