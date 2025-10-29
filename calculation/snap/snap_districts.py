import pandas as pd
import numpy as np
from policyengine_us import Microsimulation

states = ['AL', 'AK', 'AZ', 'AR', 'CA', 'CO', 'CT', 'DE', 'DC', 'FL',
          'GA', 'HI', 'ID', 'IL', 'IN', 'IA', 'KS', 'KY', 'LA', 'ME',
          'MD', 'MA', 'MI', 'MN', 'MS', 'MO', 'MT', 'NE', 'NV', 'NH',
          'NJ', 'NM', 'NY', 'NC', 'ND', 'OH', 'OK', 'OR', 'PA', 'RI',
          'SC', 'SD', 'TN', 'TX', 'UT', 'VT', 'VA', 'WA', 'WV', 'WI', 'WY']

all_results = []

for state in states:
    print(f"Processing {state}...")
    sim = Microsimulation(dataset=f"hf://policyengine/policyengine-us-data/{state}.h5")
    df = sim.calculate_dataframe(["household_id", "household_weight", "congressional_district_geoid", "snap"], map_to="household")
    df['weighted_snap'] = df['household_weight'] * df['snap']
    weighted_totals = df.groupby('congressional_district_geoid')['weighted_snap'].sum().reset_index()
    weighted_totals.rename(columns={'weighted_snap': 'total_weighted_snap'}, inplace=True)
    weighted_totals['state'] = state
    all_results.append(weighted_totals)

combined_df = pd.concat(all_results, ignore_index=True)
combined_df.to_csv('snap_by_congressional_district.csv', index=False)
print("--- Weighted SNAP Totals by Congressional District (All States) ---")
print(combined_df)
print(f"\nTotal districts: {len(combined_df)}")
print(f"Total SNAP benefits: ${combined_df['total_weighted_snap'].sum():,.0f}")
