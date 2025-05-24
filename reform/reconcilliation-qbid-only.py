# This code is a QBID-focused simplifed adaptation of Pavel's notebook at
# analysis-notebooks/us/reconciliation/2026_reconciliation.ipynb

from copy import deepcopy

import pandas as pd
import numpy as np

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform
from policyengine_us_data.datasets.cps import EnhancedCPS_2024


tcja_qbid_reform_make_permanent_dict = {
  "gov.contrib.reconciliation.qbid.in_effect": {
    "2026-01-01.2100-12-31": False
  },
  "gov.contrib.reconciliation.qbid.phase_out_rate": {
    "2026-01-01.2100-12-31": 0.0 
  },
  "gov.contrib.reconciliation.qbid.use_bdc_income": {
    "2026-01-01.2100-12-31": False
  },
  "gov.irs.deductions.qbi.max.rate": {
    "2026-01-01.2100-12-31": 0.20
  },
  "gov.irs.deductions.qbi.max.w2_wages.rate": {
    "2026-01-01.2100-12-31": 0.5
  },
  "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {
    "2026-01-01.2035-12-31": 0.25
  },
  "gov.irs.deductions.qbi.max.business_property.rate": {
    "2026-01-01.2100-12-31": 0.025
  },
  # Keep the phase out the way it was during TCJA
  "gov.irs.deductions.qbi.phase_out.length.HEAD_OF_HOUSEHOLD": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.JOINT": {
    "2026-01-01.2100-12-31": 100_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SEPARATE": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SINGLE": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SURVIVING_SPOUSE": {
    "2026-01-01.2100-12-31": 100_000.0 
  },
  # Keep the threshold at 2025 levels under TCJA 
  "gov.irs.deductions.qbi.phase_out.start.JOINT": {
    "2026-01-01.2100-12-31": 394_600
  },
  "gov.irs.deductions.qbi.phase_out.start.SINGLE": {
    "2026-01-01.2100-12-31": 197_300 
  },
  "gov.irs.deductions.qbi.phase_out.start.SEPARATE": {
    "2026-01-01.2100-12-31": 197_300 
  },
  "gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE": {
    "2026-01-01.2100-12-31": 394_600 
  },
  "gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD": {
    "2026-01-01.2100-12-31": 197_300 
  }
}

tuples = []
for year in range(2026, 2035):
    print(year)
    
    # JCT #5: Extension of deduction for qualified businesss income & permanent enhancement
    tcja_qbid_reform_make_permanent = Reform.from_dict(tcja_qbid_reform_make_permanent_dict)
    
    sim_qbid_off = Microsimulation(
        dataset=EnhancedCPS_2024,
    )
    sim_qbid_stays_on = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=tcja_qbid_reform_make_permanent)
    
    qbid_off_income = sim_qbid_off.calculate("income_tax", period=year).sum()
    qbid_stays_on_income = sim_qbid_stays_on.calculate("income_tax", period=year).sum()
    
    print(f"JCT 5 (millions): {(qbid_stays_on_income - qbid_off_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 5: extension of QBID", round(qbid_stays_on_income - qbid_off_income)))

    # JCT #6: Modification to qualified business income deduction phaseout
    phaseout_modification_reform_dict = deepcopy(tcja_qbid_reform_make_permanent_dict)
    phaseout_modification_reform_dict["gov.contrib.reconciliation.qbid.in_effect"] = {
        "2026-01-01.2100-12-31": True 
    }
    phaseout_modification_reform_dict["gov.contrib.reconciliation.qbid.phase_out_rate"] = {
        "2026-01-01.2100-12-31": 0.75
    }
    phaseout_modification_reform = Reform.from_dict(phaseout_modification_reform_dict)
    
    sim_qbid_with_new_phaseout = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=phaseout_modification_reform)
    
    qbid_with_new_phaseout_income = sim_qbid_with_new_phaseout.calculate("income_tax", period=year).sum()
    
    print(f"JCT 6 (millions): {(qbid_with_new_phaseout_income - qbid_stays_on_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 6: Modification of Phaseout", round(qbid_with_new_phaseout_income - qbid_stays_on_income)))

    # JCT #7: Increase qualified business income deduction rate to 23 percent 
    increase_to_23pct_reform_dict = deepcopy(phaseout_modification_reform_dict)
    increase_to_23pct_reform_dict["gov.irs.deductions.qbi.max.rate"] = {
        "2026-01-01.2100-12-31": 0.23
    }
    increase_to_23pct_reform = Reform.from_dict(increase_to_23pct_reform_dict)
    
    sim_qbid_with_23pct_rate = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=increase_to_23pct_reform)
    
    qbid_with_23pct_income = sim_qbid_with_23pct_rate.calculate("income_tax", period=year).sum()
    print(f"JCT 7 (millions): {(qbid_with_23pct_income - qbid_with_new_phaseout_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 7: QBID increase to 23%", round(qbid_with_23pct_income - qbid_with_new_phaseout_income)))

    # JCT #8: Modification to indexing for qualifie business income deduction
    modification_to_indexing_reform_dict = deepcopy(increase_to_23pct_reform_dict)
    
    modification_to_indexing_reform_dict["gov.irs.deductions.qbi.phase_out.start.JOINT"] = {
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
    }
    
    modification_to_indexing_reform_dict["gov.irs.deductions.qbi.phase_out.start.SINGLE"] = {
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
      }
    
    modification_to_indexing_reform_dict["gov.irs.deductions.qbi.phase_out.start.SEPARATE"] = {
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
    
    modification_to_indexing_reform_dict["gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE"] = {
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
      }
    
    modification_to_indexing_reform_dict["gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD"] = {
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
    
    modification_to_indexing_reform = Reform.from_dict(modification_to_indexing_reform_dict)
    
    sim_qbid_with_indexing_reform = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=modification_to_indexing_reform)
    
    qbid_with_indexing_reform_income = (
        sim_qbid_with_indexing_reform.calculate("income_tax", period=year).sum()
    )
    print(f"JCT 8 (millions): {(qbid_with_indexing_reform_income - qbid_with_23pct_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 8: QBID reindexing%", round(qbid_with_indexing_reform_income - qbid_with_new_phaseout_income)))


reforms_df = pd.DataFrame(tuples)
reforms_df.columns = ['year', 'jct_reform_no', 'budget_impact_mil']
reforms_df.sort_values(['year', 'jct_reform_no']).to_csv('qbid_budget_impacts.csv')
reforms_df


