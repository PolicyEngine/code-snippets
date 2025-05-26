from copy import deepcopy
import math
from __future__ import annotations
from dateutil.relativedelta import relativedelta
from pathlib import Path
import json
from typing import Dict

import pandas as pd
import numpy as np

from policyengine_us import Microsimulation
from policyengine_core.reforms import Reform
from policyengine_us_data.datasets.cps import EnhancedCPS_2024


# ---------------------------------------------------------------------
# 1. Load historical C-CPI-U (FRED series SUUR0000SA0)
# ---------------------------------------------------------------------
CPI_PATH = Path("/mnt/c/devl/data/pe/FRED-SUUR0000SA0.csv")
cpi_df = pd.read_csv(CPI_PATH)

cpi_df["observation_date"] = pd.to_datetime(cpi_df["observation_date"])
cpi = cpi_df.set_index("observation_date")["SUUR0000SA0"]

# ---------------------------------------------------------------------
# 2. Helper – Sep→Aug average for a *calendar* year
# ---------------------------------------------------------------------
def chained_avg(cal_year: int, series: pd.Series) -> float:
    start = pd.Timestamp(cal_year - 1, 9, 1)
    end = pd.Timestamp(cal_year, 8, 31)
    return series.loc[start:end].mean()


# ---------------------------------------------------------------------
# 3. Statutory constants
# ---------------------------------------------------------------------
BASE_AVG_2017 = chained_avg(2017, cpi)  # current law
BASE_AVG_2016 = chained_avg(2016, cpi)  # H.R. 1
BASE_DOLLARS = 157_500                  # §199A(e)(2)(A)

# ---------------------------------------------------------------------
# 4. Historical thresholds 2018-2025
# ---------------------------------------------------------------------
hist_rows: list[dict[str, int]] = []
for tax_year in range(2018, 2026):
    col_avg = chained_avg(tax_year - 1, cpi)

    orig = math.floor(BASE_DOLLARS * col_avg / BASE_AVG_2017 / 100) * 100
    hr1 = math.floor(BASE_DOLLARS * col_avg / BASE_AVG_2016 / 100) * 100

    hist_rows.append({"tax_year": tax_year, "orig_199A": orig, "hr1_rebase": hr1})

threshold_df = pd.DataFrame(hist_rows).set_index("tax_year")

# ---------------------------------------------------------------------
# 5. Forecast monthly C-CPI-U (steady 2 % annual inflation)
# ---------------------------------------------------------------------
INFL_RATE = 0.02        # 2 % per year
HORIZON_YEARS = 10

last_date = cpi.index.max()
last_val = cpi.iloc[-1]

months_needed = 12 * (HORIZON_YEARS + 2)     # +2 to reach Sep-window
growth_factor = 1 + INFL_RATE / 12

future_dates = [
    last_date + relativedelta(months=i + 1) for i in range(months_needed)
]
future_vals = [last_val * growth_factor ** (i + 1) for i in range(months_needed)]

# ★ Name the Series so concat returns a *Series*, not a DataFrame
cpi_fcast = pd.Series(
    future_vals,
    index=future_dates,
    name="SUUR0000SA0",
)

cpi_extended = pd.concat([cpi, cpi_fcast]).sort_index()

# ---------------------------------------------------------------------
# 6. Helper that now uses the extended series
# ---------------------------------------------------------------------
def chained_avg_ext(cal_year: int) -> float:
    return chained_avg(cal_year, cpi_extended)


# ---------------------------------------------------------------------
# 7. Future thresholds 2026-2035
# ---------------------------------------------------------------------
future_rows: list[dict[str, int]] = []
for tax_year in range(2026, 2026 + HORIZON_YEARS):
    col_avg = chained_avg_ext(tax_year - 1)

    orig = math.floor(BASE_DOLLARS * col_avg / BASE_AVG_2017 / 100) * 100
    hr1 = math.floor(BASE_DOLLARS * col_avg / BASE_AVG_2016 / 100) * 100

    future_rows.append({"tax_year": tax_year, "orig_199A": orig, "hr1_rebase": hr1})

future_df = pd.DataFrame(future_rows).set_index("tax_year")

# ---------------------------------------------------------------------
# 8. Combined table (2018-2035)
# ---------------------------------------------------------------------
all_thresholds = pd.concat([threshold_df, future_df])
all_thresholds

def build_qbi_threshold_mapping(
    projection: pd.Series,
    *,
    single_statuses: tuple[str, ...] = (
        "SINGLE",
        "HEAD_OF_HOUSEHOLD",
        "SEPARATE",
    ),
    double_statuses: tuple[str, ...] = ("JOINT", "SURVIVING_SPOUSE"),
) -> Dict[str, Dict[str, int]]:
    """
    Convert a 2026-2035 projection (index = years, values = thresholds)
    into the PolicyEngine date-slice JSON expected for §199A thresholds.

    * `projection` – a Series whose index is the tax year (int) and whose
      values are the **single-filer** dollar amounts.
    * `single_statuses` – filing statuses that use the projection as-is.
    * `double_statuses` – statuses that get twice the single amount.
    """
    def year_slices(series: pd.Series) -> Dict[str, int]:
        last_year = int(series.index.max())
        out: Dict[str, int] = {}
        for yr, val in series.items():
            end_year = 2100 if yr == last_year else yr
            out[f"{yr}-01-01.{end_year}-12-31"] = int(val)  # ← integer!
        return out

    thresholds: Dict[str, Dict[str, int]] = {}

    for status in single_statuses:
        key = f"gov.irs.deductions.qbi.phase_out.start.{status}"
        thresholds[key] = year_slices(projection)

    for status in double_statuses:
        key = f"gov.irs.deductions.qbi.phase_out.start.{status}"
        thresholds[key] = year_slices(projection * 2)

    return thresholds

# ------------------------------------------------------------------
# Example usage
# ------------------------------------------------------------------
# Pick which projection you want:
#   single_series = future_df["orig_199A"]     # current law
#   single_series = future_df["hr1_rebase"]    # H.R. 1
single_series = future_df["orig_199A"]
tcja_thresholds_mapping = build_qbi_threshold_mapping(single_series)

hr1_series = future_df["hr1_rebase"]
hr1_thresholds_mapping = build_qbi_threshold_mapping(hr1_series)

# Base Reform for making TJIA Permanent, keeping everything else the same ---------

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
}

tcja_qbid_reform_make_permanent_dict.update(tcja_thresholds_mapping)
print(json.dumps(tcja_qbid_reform_make_permanent_dict, indent=2))

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

    # NOTE: doing some backing in to JCT numbers
    #self_employment_income = sim_qbid_stays_on.calculate("self_employment_income", year).values
    #qbi = sim_qbid_stays_on.calculate("qualified_business_income", year).values
    #np.mean(qbi)
    ## The caching is brutal
    #sim_qbid_stays_on = Microsimulation(
    #    dataset=EnhancedCPS_2024,
    #    reform=tcja_qbid_reform_make_permanent)

    #sim_qbid_off.set_input("self_employment_income", year, self_employment_income / 2)
    #sim_qbid_stays_on.set_input("self_employment_income", year, self_employment_income / 2)
    #qbi2 = sim_qbid_stays_on.calculate("qualified_business_income", year).values
    #np.mean(qbi2)

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

    # JCT #8: Modification to indexing for qualified business income deduction
    modification_to_indexing_reform_dict = deepcopy(increase_to_23pct_reform_dict)
    modification_to_indexing_reform_dict.update(hr1_thresholds_mapping)
   
    modification_to_indexing_reform = Reform.from_dict(modification_to_indexing_reform_dict)
    
    sim_qbid_with_indexing_reform = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=modification_to_indexing_reform)
    
    qbid_with_indexing_reform_income = (
        sim_qbid_with_indexing_reform.calculate("income_tax", period=year).sum()
    )
    print(f"JCT 8 (millions): {(qbid_with_indexing_reform_income - qbid_with_23pct_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 8: QBID reindexing%",
                   round(qbid_with_indexing_reform_income - qbid_with_23pct_income)))

    # JCT #9: BDC income qualifies for the qualified business income deduction
    bdc_income_qualifies_reform_dict = deepcopy(modification_to_indexing_reform_dict)
    bdc_income_qualifies_reform_dict["gov.contrib.reconciliation.qbid.use_bdc_income"] = {
        "2026-01-01.2100-12-31": True
    }
    bdc_income_qualifies_reform = Reform.from_dict(bdc_income_qualifies_reform_dict)
    
    sim_qbid_with_bdc_reform = Microsimulation(
        dataset=EnhancedCPS_2024,
        reform=bdc_income_qualifies_reform)

    # NOTE: doing some backing in
    # bdc_amount = sim_qbid_with_bdc_reform.calculate("qualified_bdc_income", year).values
    # bdc_amount2 = bdc_amount * 7 
    # np.mean(bdc_amount2)
    # sim_qbid_with_bdc_reform.set_input("qualified_bdc_income", year, bdc_amount2)

    qbid_with_bdc_reform_income = (
        sim_qbid_with_bdc_reform.calculate("income_tax", period=year).sum()
    )
    print(f"JCT 9 (millions): {(qbid_with_bdc_reform_income - qbid_with_indexing_reform_income) / 1E6:,.0f}")
    tuples.append((year, "JCT 9: QBID with BDC income included:",
                   round(qbid_with_bdc_reform_income - qbid_with_indexing_reform_income)))

reforms_df = pd.DataFrame(tuples)
reforms_df.columns = ['year', 'jct_reform_no', 'budget_impact_mil']
reforms_df.sort_values(['year', 'jct_reform_no']).to_csv('qbid_budget_impacts.csv')
reforms_df


