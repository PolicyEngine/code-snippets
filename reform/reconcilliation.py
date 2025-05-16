# This code was slighly adapted from Pavel's notebook to run with local installations
# of the various repos

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform
from policyengine_us_data.datasets.cps import EnhancedCPS_2024

import pandas as pd


baseline_branching_reform = Reform.from_dict(
    {
        "gov.simulation.branch_to_determine_itemization": {
            "2026-01-01.2100-12-31": True
        },
    },
    country_id="us",
)

qbid_reform = Reform.from_dict({
  "gov.irs.deductions.qbi.max.rate": {
    "2026-01-01.2100-12-31": 0.22
  },
  "gov.irs.deductions.qbi.max.w2_wages.rate": {
    "2026-01-01.2100-12-31": 0.5
  },
  "gov.contrib.reconciliation.qbid.in_effect": {
    "2026-01-01.2100-12-31": True
  },
  "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {
    "2026-01-01.2035-12-31": 0.25
  },
  "gov.irs.deductions.qbi.phase_out.start.JOINT": {
    "2026-01-01.2026-12-31": 400600,
    "2027-01-01.2027-12-31": 410500,
    "2028-01-01.2028-12-31": 419000,
    "2029-01-01.2029-12-31": 427350,
    "2030-01-01.2030-12-31": 435900,
    "2031-01-01.2031-12-31": 444500,
    "2032-01-01.2032-12-31": 453250,
    "2033-01-01.2033-12-31": 462200,
    "2034-01-01.2034-12-31": 471400,
    "2035-01-01.2036-12-31": 480700
  },
  "gov.irs.deductions.qbi.phase_out.start.SINGLE": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217900,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  },
  "gov.irs.deductions.qbi.phase_out.start.SEPARATE": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217950,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  },
  "gov.irs.deductions.qbi.max.business_property.rate": {
    "2026-01-01.2100-12-31": 0.025
  },
  "gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE": {
    "2026-01-01.2026-12-31": 400600,
    "2027-01-01.2027-12-31": 410500,
    "2028-01-01.2028-12-31": 419000,
    "2029-01-01.2029-12-31": 427350,
    "2030-01-01.2030-12-31": 435900,
    "2031-01-01.2031-12-31": 444500,
    "2032-01-01.2032-12-31": 453250,
    "2033-01-01.2033-12-31": 462200,
    "2034-01-01.2034-12-31": 471400,
    "2035-01-01.2036-12-31": 480700
  },
  "gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217950,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  }
}, country_id="us")

qbid_reform.__name__ = "QBID Reform"

years = range(2025, 2028)
reforms = [qbid_reform]

results_df = pd.DataFrame(columns=["Year", "Reform", "Impact (billions)"])


def calculate_stacked_budgetary_impact(reforms, year, results_df):
    """
    Calculate the incremental budgetary impact of each reform when applied sequentially.
    
    Parameters:
    reforms - List of reform objects
    year - Year to perform calculation for
    results_df - DataFrame to append results to
    
    Returns:
    Updated results_df with new rows
    """
    # Start with baseline
    baseline = Microsimulation(
        reform=baseline_branching_reform, 
        dataset=EnhancedCPS_2024
    )
    baseline_income = baseline.calculate("income_tax", map_to="household", period=year).sum()
    
    previous_income = baseline_income
    cumulative_reform = baseline_branching_reform
    
    print(f"\nCalculating sequential reforms for Year: {year}...")
    
    for i, reform in enumerate(reforms):
        # Add the current reform to the cumulative reform
        if i == 0:
            # For the first reform, start with baseline_branching_reform and add the current reform
            cumulative_reform = (baseline_branching_reform, reform)
        else:
            # For subsequent reforms, add to the already existing cumulative reform
            cumulative_reform = (cumulative_reform, reform)
        
        # Calculate with the cumulative reform
        reformed = Microsimulation(
            reform=cumulative_reform, 
            dataset=EnhancedCPS_2024
        )
        reformed_income = reformed.calculate("income_tax", map_to="household", period=year).sum()
        
        # Calculate impact compared to the previous state
        budgetary_impact = reformed_income - previous_income
        impact_billions = budgetary_impact / 1e9
        
        print(f"  {reform.__name__}: ${impact_billions:.3f} billion")
        
        # Add to results
        new_row = {
            "Year": year,
            "Reform": reform.__name__,
            "Impact (billions)": impact_billions
        }
        results_df = pd.concat([results_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Update for next iteration
        previous_income = reformed_income
    
    return results_df

# Run calculations for each year
for year in years:
    results_df = calculate_stacked_budgetary_impact(reforms, year, results_df)

