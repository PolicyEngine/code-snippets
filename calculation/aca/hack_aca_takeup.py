import numpy as np
import h5py
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.enums import Enum

TARGET_PERSONS = 21_800_000
SEED = 42
OUTPUT_PATH = Path("enhanced_cps_2024_aca_hack.h5")

sim0 = Microsimulation()
takeup = sim0.calculate("takes_up_aca_if_eligible", period=2025).values
weight = sim0.calculate("tax_unit_weight", period=2025).values
total_weight = weight.sum()
current_true_weight = weight[takeup == 1].sum()

false_indices = np.where(takeup == 0)[0]
rng = np.random.default_rng(SEED)
rng.shuffle(false_indices)

# Precompute cumulative weights for the shuffled false indices
cum_weights = np.cumsum(weight[false_indices])


def flip_takeup(additional_weight_needed):
    if additional_weight_needed <= 0:
        return takeup.copy().astype(bool)
    n_flips = np.searchsorted(cum_weights, additional_weight_needed, side="left") + 1
    n_flips = min(n_flips, len(false_indices))
    new_takeup = takeup.copy().astype(bool)
    new_takeup[false_indices[:n_flips]] = True
    return new_takeup


def count_aca_persons(new_takeup):
    sim = Microsimulation()
    sim.set_input("takes_up_aca_if_eligible", 2024, new_takeup)
    has_ptc = sim.calculate("aca_ptc", period=2025, map_to="person") > 0
    eligible = sim.calculate("is_aca_ptc_eligible", period=2025)
    return float((has_ptc * eligible).sum()), sim


# Binary search on takeup rate
lo, hi = 0.0, 1.0 - (current_true_weight / total_weight)
print(f"Searching for takeup rate that yields {TARGET_PERSONS:,.0f} ACA PTC persons...")

best_sim = None
for iteration in range(20):
    mid = (lo + hi) / 2
    additional = mid * total_weight
    new_takeup = flip_takeup(additional)
    rate = np.average(new_takeup, weights=weight)
    persons, sim = count_aca_persons(new_takeup)
    print(f"  iter {iteration}: rate={rate:.6f}, persons={persons:,.0f}")

    if abs(persons - TARGET_PERSONS) < 100_000:
        best_sim = sim
        best_takeup = new_takeup
        best_rate = rate
        best_persons = persons
        break
    elif persons < TARGET_PERSONS:
        lo = mid
    else:
        hi = mid
        best_sim = sim
        best_takeup = new_takeup
        best_rate = rate
        best_persons = persons

print(f"\nFound rate: {best_rate:.6f} -> {best_persons:,.0f} persons")

# Write h5 with the found takeup
sim_write = Microsimulation()
sim_write.set_input("takes_up_aca_if_eligible", 2024, best_takeup)

data = {}
for variable in sim_write.tax_benefit_system.variables:
    var_meta = sim_write.tax_benefit_system.variables[variable]
    if var_meta.formulas:
        continue
    holder_periods = sim_write.get_holder(variable).get_known_periods()
    if not holder_periods:
        continue
    data[variable] = {}
    for time_period in holder_periods:
        values = sim_write.get_holder(variable).get_array(time_period)
        if var_meta.value_type in (Enum, str):
            if hasattr(values, "decode_to_str"):
                values = values.decode_to_str().astype("S")
            else:
                values = values.astype("S")
        else:
            values = np.array(values)
        if values is not None:
            data[variable][time_period] = values

holder = sim_write.get_holder("person_id")
if holder.get_known_periods():
    data["person_id"] = {}
    for tp in holder.get_known_periods():
        data["person_id"][tp] = np.array(holder.get_array(tp))

with h5py.File(OUTPUT_PATH, "w") as f:
    for variable, periods in data.items():
        grp = f.create_group(variable)
        for period, values in periods.items():
            grp.create_dataset(str(period), data=values)

print(f"Written to {OUTPUT_PATH}")

# Verify
sim2 = Microsimulation(dataset=str(OUTPUT_PATH))
has_ptc = sim2.calculate("aca_ptc", period=2025, map_to="person") > 0
eligible = sim2.calculate("is_aca_ptc_eligible", period=2025)
verified_persons = float((has_ptc * eligible).sum())
verified_rate = sim2.calculate("takes_up_aca_if_eligible", period=2025).mean()
print(f"Verified: rate={float(verified_rate):.6f}, persons={verified_persons:,.0f}")

# Before/after comparison
has_ptc_orig = sim0.calculate("aca_ptc", period=2025, map_to="person") > 0
eligible_orig = sim0.calculate("is_aca_ptc_eligible", period=2025)
orig_persons = float((has_ptc_orig * eligible_orig).sum())
print(f"\nBEFORE: {orig_persons:,.0f} persons with ACA PTC")
print(f"AFTER:  {verified_persons:,.0f} persons with ACA PTC")
