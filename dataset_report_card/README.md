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

Edit `common.py` to set the two dataset paths:

```python
# Dataset A — the baseline (typically the production enhanced CPS from HF)
PATH_A = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"

# Dataset B — the candidate (HF or local)
PATH_B = "hf://policyengine/test/apr/national/US.h5"
# PATH_B = "/home/you/devl/policyengine-us-data/local_area_build/national/US.h5"
```

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
