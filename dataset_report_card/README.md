# Dataset Report Card

Compares two PolicyEngine US datasets side-by-side across point-target checks, range checks, and state-level checks (ACA PTC, Medicaid enrollment), then produces a summary report showing which dataset wins on each metric.

## Loading datasets

Datasets can be loaded from Hugging Face or from a local file path:

```python
from policyengine_us import Microsimulation

# From Hugging Face (hf://org/repo/path/to/file.h5)
sim = Microsimulation(dataset="hf://policyengine/test/apr/national/US.h5")

# From a local build
sim = Microsimulation(dataset="/path/to/local_area_build/national/US.h5")
```

## Configuration

`PATH_A` is a Hugging Face URL and lives in `common.py`. `PATH_B` and the state-check calibration targets both derive from a local `policyengine-us-data` checkout, controlled by one env var:

```bash
export POLICYENGINE_US_DATA_ROOT=/path/to/policyengine-us-data
```

If unset, it defaults to `~/policyengine-us-data`. The expected layout under the root is:

```
policyengine-us-data/
├── local_area_build/national/US.h5              # becomes PATH_B
└── policyengine_us_data/storage/calibration_targets/
    ├── aca_spending_and_enrollment_2024.csv
    └── medicaid_enrollment_2024.csv
```

To point Dataset B at a Hugging Face URL instead, edit `PATH_B` in `common.py` directly.

## Running

```bash
./run.sh
```

This runs three steps:

1. **step1_dataset_a.py** — runs Dataset A checks in short-lived subprocesses, writes `results_a.json`
2. **step2_dataset_b.py** — runs Dataset B checks in short-lived subprocesses, writes `results_b.json`
3. **step3_report.py** — reads both result files and prints a comparison report (pass `--text` to also write `report.txt`)

The subprocess pattern is deliberate. Each check gets its own `Microsimulation`, then the process exits so memory is returned to the OS before the next check starts.
If one subprocess is killed or errors, that check is recorded as an error in the report and the remaining checks continue.

## Checks

| Type | What it checks |
|------|---------------|
| Point-target | SNAP, Social Security, SSI, employment income, AGI, income tax, EITC, CTC, tips, rent, real estate taxes, population/household counts |
| Range | Floors and ceilings for income tax, employment income, household/person weights, poverty rate, citizenship pct |
| Consistency | Direct invariants from the integration tests, such as `UNDOCUMENTED == SSN NONE` |
| State-level | ACA PTC spending by state, Medicaid enrollment by state |

The state checks now run for both datasets, so ACA and Medicaid failures appear in the scorecard instead of being silently omitted when only one side was computed.
