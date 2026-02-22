#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR/.."
DEST_DATA_DIR="$PROJECT_ROOT/data"

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <dataset_name>"
    echo "Example: $0 tgbl-wiki"
    exit 1
fi

DATASET_NAME="$1"

mkdir -p "$DEST_DATA_DIR"

uv run --no-project --with py-tgb --with torch --with numpy python - <<EOF
import numpy as np
from tgb.linkproppred.dataset import LinkPropPredDataset
from tgb.nodeproppred.dataset import NodePropPredDataset

name, dest = "$DATASET_NAME", "$DEST_DATA_DIR"
print(f"Loading TGB dataset: {name}")

# TODO(kuba): Handle splits and negatives
ds_cls = NodePropPredDataset if name.startswith('tgbn-') else LinkPropPredDataset
data = ds_cls(name=name).full_data

cols = [data['sources'], data['destinations'], data['timestamps']]
header = "src,dst,time"

if data['edge_feat'] is not None:
    cols.append(data['edge_feat'])
    header += "," + ",".join(f"msg_{i}" for i in range(data['edge_feat'].shape[1]))

out_path = f"{dest}/{name}.csv"
np.savetxt(out_path, np.column_stack(cols), delimiter=",", header=header, comments='', fmt='%g')
print(f"Exported {name} with {len(cols[0])} edges to {out_path}")
EOF
