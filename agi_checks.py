import numpy as np
import pandas as pd
from policyengine_us import Microsimulation

sim = Microsimulation(dataset="hf://policyengine/test/apr/states/NY.h5")
YEAR = 2024

agi_mdf = sim.calculate("adjusted_gross_income", period=YEAR, map_to="household")
weight = sim.calculate("household_weight", period=YEAR).values

df = pd.DataFrame({"agi": agi_mdf.values, "weight": weight, "tax": 6}) #tax})
df["agi_decile"] = agi_mdf.decile_rank().values.astype(int)

print("=== AGI Distribution by Decile (weighted) ===")
for d in range(1, 11):
    s = df[df["agi_decile"] == d]
    if len(s) == 0:
        print(f"Decile {d:2d}: (empty)")
        continue
    avg_agi = np.average(s["agi"], weights=s["weight"])
    min_agi = s["agi"].min()
    max_agi = s["agi"].max()
    total_tax = (s["tax"] * s["weight"]).sum()
    print(
        f"Decile {d:2d}: AGI ${min_agi:,.0f} – ${max_agi:,.0f}, avg ${avg_agi:,.0f}"
    )


#sim = Microsimulation(dataset="/home/baogorek/devl/temp/policyengine-us-data/local_area_build/states/WA.h5")
#tax = sim.calculate("wa_millionaires_tax", period=YEAR, map_to="household").values
#looking strictly at the AGI distribution in a vacuum.
#
# Check for families with young children in decile 1
print("\n=== Young Children in Decile 1 ===")
person_df = sim.calculate_dataframe(
    ["person_id", "household_id", "age", "is_child"],
    period=YEAR,
)
hh_ids = sim.calculate("household_id", period=YEAR, map_to="household").values
hh_df = pd.DataFrame({"household_id": hh_ids, "agi_decile": df["agi_decile"].values})

merged = person_df.merge(hh_df, on="household_id")
decile1 = merged[merged["agi_decile"] == 1]
young = decile1[decile1["age"] < 6]

print(f"Total persons in decile 1: {len(decile1)}")
print(f"Young children (age < 6) in decile 1: {len(young)}")
print(f"Households with young children in decile 1: {young['household_id'].nunique()}")
if len(young) > 0:
    print("\nSample young children in decile 1:")
    print(young[["person_id", "household_id", "age", "is_child"]].head(20).to_string())
else:
    print("No young children found in decile 1.")
