import numpy as np
import pandas as pd
from microdf import MicroDataFrame
from policyengine_us import Microsimulation

from agi_checks_helpers import summarize_young_children


DATASET = "hf://policyengine/test/apr/states/NY.h5"
STATE = DATASET.rsplit("/", 1)[-1].removesuffix(".h5")
YEAR = 2024


def summarize_agi(series):
    mask = (series.weights > 0).to_numpy() & np.isfinite(series.values)
    series = series[mask]

    if len(series) == 0:
        return {
            "households": 0,
            "weighted_households": 0.0,
            "min_agi": np.nan,
            "median_agi": np.nan,
            "mean_agi": np.nan,
            "max_agi": np.nan,
        }

    return {
        "households": int(len(series)),
        "weighted_households": float(series.weights.sum()),
        "min_agi": float(series.min()),
        "median_agi": float(series.median()),
        "mean_agi": float(series.mean()),
        "max_agi": float(series.max()),
    }


fmt_currency = lambda value: "nan" if pd.isna(value) else f"${value:,.0f}"

sim = Microsimulation(dataset=DATASET)

agi = sim.calculate("adjusted_gross_income", period=YEAR, map_to="household")
household_weight = sim.calculate("household_weight", period=YEAR, map_to="household")
district = sim.calculate("congressional_district_geoid", period=YEAR, map_to="household")

# `calculate_dataframe(..., map_to="household")` is pulling mismatched weights
# here, so build the MicroDataFrame directly from household-mapped series.
household_agi = MicroDataFrame(
    pd.DataFrame(
        {
            "agi": agi.values,
            "district": district.values,
        }
    ),
    weights=household_weight.values,
)

agi = household_agi["agi"]

state_summary = summarize_agi(agi)

print(f"=== {STATE} Adjusted Gross Income Summary (weighted) ===")
print(f"Households: {state_summary['households']:,}")
print(f"Weighted households: {state_summary['weighted_households']:,.0f}")
print(f"Min AGI: {fmt_currency(state_summary['min_agi'])}")
print(f"Median AGI: {fmt_currency(state_summary['median_agi'])}")
print(f"Mean AGI: {fmt_currency(state_summary['mean_agi'])}")
print(f"Max AGI: {fmt_currency(state_summary['max_agi'])}")

agi_deciles = agi.decile_rank()
decile_rows = []

print("\n=== AGI Distribution by Decile (weighted) across the state ===")
for decile in range(1, 11):
    decile_slice = agi[agi_deciles == decile]
    decile_rows.append({"decile": decile, **summarize_agi(decile_slice)})

decile_summary = pd.DataFrame(decile_rows)
formatted_deciles = decile_summary.copy()
formatted_deciles["weighted_households"] = formatted_deciles["weighted_households"].map(
    lambda value: f"{value:,.0f}"
)
for column in ["min_agi", "median_agi", "mean_agi", "max_agi"]:
    formatted_deciles[column] = formatted_deciles[column].map(fmt_currency)
print(formatted_deciles.to_string(index=False))

district_rows = []
district_geoids = sorted(pd.Series(household_agi["district"].values).dropna().astype(int).unique())

for geoid in district_geoids:
    district_slice = household_agi[household_agi["district"] == geoid]["agi"]
    district_rows.append(
        {
            "district": f"{STATE}-{geoid % 100:02d}",
            "geoid": geoid,
            **summarize_agi(district_slice),
        }
    )

district_summary = pd.DataFrame(district_rows).sort_values("geoid").reset_index(drop=True)

print("\n=== District AGI Summary (weighted) ===")
formatted_districts = district_summary.copy()
formatted_districts["weighted_households"] = formatted_districts["weighted_households"].map(
    lambda value: f"{value:,.0f}"
)
for column in ["min_agi", "median_agi", "mean_agi", "max_agi"]:
    formatted_districts[column] = formatted_districts[column].map(fmt_currency)
print(formatted_districts.to_string(index=False))

district_weight_sum = district_summary["weighted_households"].sum()
district_mean = (
    district_summary["mean_agi"] * district_summary["weighted_households"]
).sum() / district_weight_sum

print("\n=== Sanity Checks ===")
print(
    "District weights sum to state weights:",
    np.isclose(district_weight_sum, state_summary["weighted_households"]),
)
print(
    "District weighted mean recomposes to state mean:",
    np.isclose(district_mean, state_summary["mean_agi"]),
)
print(
    "District minimum matches state minimum:",
    district_summary["min_agi"].min() == state_summary["min_agi"],
)
print(
    "District maximum matches state maximum:",
    district_summary["max_agi"].max() == state_summary["max_agi"],
)

print("\n=== Young Children in Decile 1 ===")
person_df = sim.calculate_dataframe(
    ["person_id", "household_id", "age", "is_child", "person_weight"],
    period=YEAR,
    use_weights=False,
)
hh_ids = sim.calculate("household_id", period=YEAR, map_to="household").values
hh_df = pd.DataFrame(
    {
        "household_id": hh_ids,
        "agi_decile": agi_deciles.values.astype(int),
        "household_weight": household_weight.values,
    }
)

merged = person_df.merge(hh_df, on="household_id")
decile1 = merged[merged["agi_decile"] == 1]
young = decile1[decile1["age"] < 6]
population_summary = summarize_young_children(decile1)

print(f"Projected persons in decile 1: {population_summary['persons']:,.0f}")
print(
    "Projected young children (age < 6) in decile 1: "
    f"{population_summary['young_children']:,.0f}"
)
print(
    "Projected households with young children in decile 1: "
    f"{population_summary['households_with_young_children']:,.0f}"
)
if len(young) > 0:
    print("\nSample young children in decile 1:")
    print(young[["person_id", "household_id", "age", "is_child"]].head(20).to_string())
else:
    print("No young children found in decile 1.")
