#!/bin/bash
set -e
cd "$(dirname "$0")"

export REPORT_SKIP_RANGE=1

source ~/envs/dev/bin/activate

python -c "from common import _DATA_ROOT, PATH_B, CAL_DIR; print(f'Data root: {_DATA_ROOT}'); print(f'PATH_B:    {PATH_B}'); print(f'CAL_DIR:   {CAL_DIR}')"

echo "=== Step 1/3: Dataset B checks (US.h5 — local, fail fast) ==="
python step2_dataset_b.py

echo ""
echo "=== Step 2/3: Dataset A checks (enhanced_cps_2024.h5 — HF) ==="
python step1_dataset_a.py

echo ""
echo "=== Step 3/3: Report ==="
python step3_report.py --text
