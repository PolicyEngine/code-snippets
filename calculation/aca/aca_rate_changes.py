
# ACA stuff
"""
 SLCSP ≠ ACA. SLCSP is actually part of the ACA system.

  ACA (Affordable Care Act) = The entire health insurance program, including:
  - Marketplace/exchanges (where you buy insurance)
  - Premium tax credits (subsidies)
  - Income eligibility rules
  - Coverage requirements
  - Etc.

  SLCSP (Second Lowest Cost Silver Plan) = A specific benchmark premium used within the ACA to calculate subsidies.

  Here's how it works:

  1. SLCSP is the benchmark — CMS identifies the 2nd cheapest Silver plan in each state/rating area
  2. SLCSP determines your subsidy — Your PTC is calculated based on SLCSP, not the actual plan you choose
  3. You can buy any plan — You can pick Bronze, Gold, Platinum, or any Silver plan. The subsidy stays the same.

  Example:
  - SLCSP in your area = $400/month
  - Your income qualifies you for 5% contribution = $20/month ($400 × 5%)
  - Your PTC subsidy = $380/month ($400 - $20)
  - If you buy a cheaper Bronze plan ($300/month), you pay $20 net and keep the extra $80
  - If you buy an expensive Platinum plan ($500/month), you pay $120 net (because subsidy stays at $380)

  So in the PolicyEngine code:
  - slcsp = The benchmark premium (what the subsidy calculation is based on)
  - aca_ptc = The actual subsidy payment (calculated from SLCSP and your income)

  The premium increase in the image is about SLCSP going up when tax credits expire — people lose subsidy help but the benchmark premium itself is
  climbing.
"""

import pandas as pd
import numpy as np
from policyengine_us import Microsimulation

sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/NC.h5")

target_variables = [
    "household_id", 
    "household_weight", 
    "congressional_district_geoid", 
    "aca_ptc", 
    "slcsp"
]

df = pd.DataFrame(sim.calculate_dataframe(target_variables))

target_district = 3701 
district_df = df[df["congressional_district_geoid"] == target_district].copy()

total_ptc_in_district = np.sum(district_df.household_weight.values * district_df.aca_ptc.values)

print(f"Total ACA PTC for District {target_district}: ${total_ptc_in_district:,.2f}")

from policyengine_core.reforms import Reform

reform = Reform.from_dict({
    "gov.aca.state_rating_area_cost": {
        "NC": {
            "14": {
                "2023-01-01": 2000.05,
            }
        },
        "AL": {
            "1": {
                "2023-01-01": 267.75,
            },
            "2": {
                "2023-01-01": 276.15,
            }
        }
    }
})

reform = Reform.from_dict({
    "gov.aca.state_rating_area_cost.NC.14": {
        "2023-01-01": 2000.05
    },
    "gov.aca.state_rating_area_cost.AL.1": {
        "2023-01-01": 267.75
    },
    "gov.aca.state_rating_area_cost.AL.2": {
        "2023-01-01": 276.15
    }
})

baseline = Microsimulation(dataset="hf://policyengine/policyengine-us-data/NC.h5")
reformed = Microsimulation(dataset="hf://policyengine/policyengine-us-data/NC.h5", reform=reform)



print(f"2025 SLCSP: ${baseline_slcsp:,.2f}")
print(f"2026 SLCSP (uprated): ${reformed_slcsp:,.2f}")

