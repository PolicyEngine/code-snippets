# ACA Premium Tax Credit: State-Level Multipliers (2022 to 2024)

## Background

PolicyEngine's microsimulation model uses the IRS Public Use File (PUF) as its
primary data source for tax-unit-level modeling of the Affordable Care Act's
Premium Tax Credit (PTC). The PUF reflects a specific tax year -- currently
2022 -- but policy analysis often targets a later year such as 2024. The ACA
marketplace has changed substantially between these years: some states saw
enrollment nearly double while average credit amounts shifted more modestly. A
national-level scaling factor would miss this variation entirely, so we need
state-level multipliers to bring the 2022 baseline forward to 2024.

## Why two multipliers?

The total PTC outlay in a state is roughly the product of two quantities:

1. **How many tax units receive the credit** (volume)
2. **How much each tax unit receives on average** (value)

These move independently. A state can have surging enrollment with a flat or
even declining average credit (e.g., Texas), or stable enrollment with a
rising average credit (e.g., New York). Capturing both dimensions requires two
separate multipliers per state.

### Volume multiplier (`vol_mult`)

```
vol_mult = effectuated_enrollment_2024 / effectuated_enrollment_2022
```

This ratio is applied to the **count of tax units** receiving PTCs in each
state. It accounts for coverage expansions, Medicaid unwinding spillover,
enhanced subsidies under the Inflation Reduction Act, and population shifts.

States like West Virginia (2.50x), Louisiana (2.45x), and Mississippi (2.22x)
saw dramatic enrollment growth, while DC (0.84x), Maine (1.00x), and Oregon
(1.01x) were essentially flat.

### Value multiplier (`val_mult`)

```
val_mult = avg_aptc_2024 / avg_aptc_2022
```

This ratio is applied to the **dollar amount of PTC** on each tax unit. It
reflects changes in benchmark premiums, enrollee income composition, and
plan-selection patterns.

Value multipliers are generally tighter, clustering around 1.0. The national
mean is 1.04 and the median is 1.03. DC (1.38x) and Alaska (1.25x) are
outliers on the high end; Iowa (0.85x) and Idaho (0.86x) saw notable declines.

## Data sources

Both multipliers are derived from the CMS Effectuated Enrollment Snapshots, as
compiled by KFF State Health Facts:

- **Effectuated enrollment:**
  [KFF - Effectuated Marketplace Enrollment and Financial Assistance](https://www.kff.org/affordable-care-act/state-indicator/effectuated-marketplace-enrollment-and-financial-assistance/)
- **Average APTC:**
  [KFF - Marketplace Average Premiums and Average APTC](https://www.kff.org/affordable-care-act/state-indicator/marketplace-average-premiums-and-average-advanced-premium-tax-credit-aptc/)

These tables report the average monthly effectuated enrollment and the average
monthly APTC among consumers receiving APTCs, by state and year.

### Nevada data note

KFF reports Nevada's 2022 average APTC as "NR" (Not Reported). We substituted
**$435/month** from the
[HHS/ASPE Nevada marketplace fact sheet](https://aspe.hhs.gov/sites/default/files/documents/State-Level-Data-on-the-ACA.pdf),
which draws on the same underlying CMS data.

## Output

Running `multipliers.py` produces:

- **`aca_ptc_multipliers_2022_2024.csv`** with columns:
  `state`, `enroll_2022`, `enroll_2024`, `vol_mult`, `aptc_2022`, `aptc_2024`, `val_mult`
- A printed summary table and descriptive statistics to stdout

## Usage

```bash
python multipliers.py
```

Requires `pandas`. All source data is hardcoded in the script since the CMS
enrollment snapshots are static historical releases.
