#!/bin/bash
set -e
cd "$(dirname "$0")"

source ~/envs/temp/bin/activate

echo "=== Step 1/3: Dataset A checks ==="
python step1_dataset_a.py

echo ""
echo "=== Step 2/3: Dataset B checks ==="
python step2_dataset_b.py

echo ""
echo "=== Step 3/3: Report ==="
python step3_report.py --text
