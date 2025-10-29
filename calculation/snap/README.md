# SNAP Benefits by Congressional District

![SNAP Benefits Map](snap_benefits_by_district.png)

## Overview

Survey-weighted estimates of total SNAP benefits by congressional district across all 50 states plus DC.

**Total SNAP Benefits: $69.4 billion**

## Prerequisites

```bash
uv pip install policyengine-us pandas --python ~/envs/pe/bin/python
```

## Running the Code

```bash
cd /home/baogorek/devl/code-snippets/calculation/snap
~/envs/pe/bin/python snap_districts.py
```

**Output:** `snap_by_congressional_district.csv`

## Data

- **Source:** PolicyEngine test repository (hf://policyengine/test) with corrected district assignments
- **Districts:** 436 congressional districts
- **Variables:** household_id, household_weight, congressional_district_geoid, state_fips, snap
- **Range:** $26M - $475M per district

## Results

### Top States by SNAP Benefits

| State FIPS | State | Total Benefits |
|------------|-------|----------------|
| 6          | CA    | $10.9B         |
| 36         | NY    | $5.5B          |
| 48         | TX    | $5.1B          |
| 12         | FL    | $4.3B          |
| 17         | IL    | $3.5B          |

### Top Districts by SNAP Benefits

| District | State FIPS | Total Benefits |
|----------|------------|----------------|
| 1502     | 15 (HI)    | $475M          |
| 3615     | 36 (NY)    | $463M          |
| 3613     | 36 (NY)    | $412M          |
| 621      | 6 (CA)     | $404M          |
| 1501     | 15 (HI)    | $403M          |

## Visualization

For map visualization code, see: `/home/baogorek/devl/Congressional-Hackathon-2025/PolicyEngine/plot_snap_impacts.py`
