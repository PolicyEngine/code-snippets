import pandas as pd
import sys

log_path = (
    "/home/baogorek/devl/temp/policyengine-us-data/"
    "policyengine_us_data/storage/calibration/calibration_log.csv"
)
if len(sys.argv) > 1:
    log_path = sys.argv[1]

df = pd.read_csv(log_path)

# Total population: sum district-level age bin targets only (cd_*)
# Excludes filer-count-by-AGI-bracket person_counts and state-level targets
pop = df[df["target_name"].str.match(r"cd_\d+/person_count/\[age<")]
pop_by_epoch = pop.groupby("epoch").agg(
    estimate=("estimate", "sum"),
    target=("target", "sum"),
).assign(
    rel_error=lambda x: (x["estimate"] - x["target"]) / x["target"],
)

# AGI: sum district-level targets only (cd_*), like population
agi = df[df["target_name"].str.match(r"cd_\d+/adjusted_gross_income/")]
agi_by_epoch = agi.groupby("epoch").agg(
    estimate=("estimate", "sum"),
    target=("target", "sum"),
).assign(
    rel_error=lambda x: (x["estimate"] - x["target"]) / x["target"],
)

fmt_T = lambda v: f"${v / 1e12:.2f}T"
fmt_M = lambda v: f"{v / 1e6:.2f}M"
fmt_pct = lambda v: f"{v:+.2%}"

print("=== Total Population (sum over districts x age bins) ===")
print(f"{'Epoch':>8}  {'Estimate':>14}  {'Target':>14}  {'Rel Error':>12}")
for epoch, row in pop_by_epoch.iterrows():
    print(
        f"{epoch:>8.0f}  {fmt_M(row['estimate']):>14}  "
        f"{fmt_M(row['target']):>14}  {fmt_pct(row['rel_error']):>12}"
    )

print()
print("=== Adjusted Gross Income (national, filers) ===")
print(f"{'Epoch':>8}  {'Estimate':>14}  {'Target':>14}  {'Rel Error':>12}")
for epoch, row in agi_by_epoch.iterrows():
    print(
        f"{epoch:>8.0f}  {fmt_T(row['estimate']):>14}  "
        f"{fmt_T(row['target']):>14}  {fmt_pct(row['rel_error']):>12}"
    )


# --------- sim.calculate check ------------------------------------
from pathlib import Path
from policyengine_us import Microsimulation
import numpy as np

basepath = Path('/home/baogorek/devl/temp/policyengine-us-data/local_area_build/states/')

population_dict = {}
agi_dict = {}

for h5 in sorted(basepath.glob("*.h5")):
    state = h5.stem
    sim = Microsimulation(dataset=str(h5))
    weights = sim.calculate("person_weight").values
    population_dict[state] = weights.sum()
    agi = sim.calculate("adjusted_gross_income", map_to="household")
    agi_dict[state] = agi.sum()

print("sim.calculate metrics")
print(f"Total population (M): {sum(population_dict.values()) / 1e6:.2f}")
print(f"Total AGI (T): {sum(agi_dict.values()) / 1e12:.2f}")

# --------- RI poverty / AGI diagnostic --------------------------------
basepath = Path('/home/baogorek/devl/temp/policyengine-us-data/local_area_build/states/')
state_path = basepath / "MI.h5"
sim = Microsimulation(dataset=str(state_path))

import pandas as pd

entity_rel = pd.DataFrame(sim.calculate_dataframe(
    ['tax_unit_household_id', 'tax_unit_id'], map_to="tax_unit"
))
hh_df = pd.DataFrame(sim.calculate_dataframe(
    ['household_id', 'household_weight'], map_to="household"
))
tax_df = pd.DataFrame(sim.calculate_dataframe(
    ['tax_unit_id', 'adjusted_gross_income', 'tax_unit_is_filer'], map_to="tax_unit"
))


agi_p = sim.calculate("adjusted_gross_income", map_to="person")
age = sim.calculate("age")
poverty = sim.calculate("in_poverty", map_to="person")
w = agi_p.weights.values
a = agi_p.values
pov = poverty.values
is_child = age.values < 18

print()
print("=== AGI Distribution (weighted) ===")
print(f"Total pop: {w.sum():,.0f}")
print(f"Total AGI: ${(a * w).sum():,.0f}")
print(f"Mean AGI (weighted): ${np.average(a, weights=w):,.0f}")
print(f"Records: {len(a):,}")
print()

brackets = [
    (-np.inf, 0), (0, 1), (1, 15000), (15000, 50000),
    (50000, 100000), (100000, 200000), (200000, 500000),
    (500000, 1e6), (1e6, np.inf),
]
print(f"{'AGI Bracket':>25}  {'Records':>8}  {'Wt Pop':>12}  {'Wt Share':>10}  {'Wt AGI':>15}")
for lo, hi in brackets:
    mask = (a >= lo) & (a < hi)
    cnt = mask.sum()
    wpop = w[mask].sum()
    wagi = (a[mask] * w[mask]).sum()
    if lo == -np.inf:
        label = "< $0"
    elif hi == np.inf:
        label = f">= ${lo:,.0f}"
    else:
        label = f"${lo:,.0f} - ${hi:,.0f}"
    print(f"{label:>25}  {cnt:>8,}  {wpop:>12,.0f}  {wpop/w.sum():>10.1%}  ${wagi:>14,.0f}")

print()
print("=== Poverty rate by AGI bracket ===")
pov_brackets = [(-np.inf, 0), (0, 1), (1, 15000), (15000, 50000), (50000, 100000), (100000, np.inf)]
for lo, hi in pov_brackets:
    mask = (a >= lo) & (a < hi)
    if w[mask].sum() == 0:
        continue
    pov_rate = np.average(pov[mask], weights=w[mask])
    if lo == -np.inf:
        label = "< $0"
    elif hi == np.inf:
        label = f">= ${lo:,.0f}"
    else:
        label = f"${lo:,.0f}-${hi:,.0f}"
    print(f"  {label:>25}: poverty={pov_rate:.1%}  (pop={w[mask].sum():,.0f})")

print()
print("=== Poverty: children vs adults ===")
print(f"  Children: {np.average(pov[is_child], weights=w[is_child]):.1%}  (pop={w[is_child].sum():,.0f})")
print(f"  Adults:   {np.average(pov[~is_child], weights=w[~is_child]):.1%}  (pop={w[~is_child].sum():,.0f})")

print()
print("=== Weight distribution ===")
print(f"  Weight > 100: {(w > 100).sum():,} records, pop share: {w[w > 100].sum()/w.sum():.1%}")
print(f"  Weight < 10:  {(w < 10).sum():,} records, pop share: {w[w < 10].sum()/w.sum():.1%}")
print(f"  Zeros/neg:    {(w <= 0).sum():,} records")
