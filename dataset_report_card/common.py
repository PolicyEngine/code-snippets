"""Shared config, helpers, and check definitions for the report card."""

from pathlib import Path
import numpy as np
import pandas as pd

# ── Config ────────────────────────────────────────────────────
PATH_A = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
PATH_B = "/home/baogorek/devl/temp/policyengine-us-data/local_area_build/national/US.h5"
LABEL_A = "enhanced_cps_2024.h5"
LABEL_B = "US.h5"
VERBOSE = False
USE_COLOR = True

CAL_DIR = Path(__file__).resolve().parents[2] / "temp" / "policyengine-us-data" / "policyengine_us_data" / "storage" / "calibration_targets"

# National target values both calibration paths target. Sourced from
# policy_data.db (target_overview, active=1, reform_id=0, period=2024,
# geo_level='national'). These match loss.py's HARD_CODED_TOTALS,
# MEDICAID_*_TARGETS, ACA_*_TARGETS, RETIREMENT_CONTRIBUTION_TARGETS,
# and get_beneficiary_paid_medicare_part_b_premiums_target(2024).
HARD_CODED_TOTALS = {
    # Hardcoded national dollar totals (legacy keys preserved)
    "tip_income": 38e9 * 1.4,
    "real_estate_taxes": 500e9,
    "rent": 735e9,
    # Healthcare
    "health_insurance_premiums_without_medicare_part_b": 385e9,
    "other_medical_expenses": 278e9,
    "medicare_part_b_premium": 112e9,
    "over_the_counter_health_expenses": 72e9,
    # Family / SPM
    "child_support_expense": 33e9,
    "child_support_received": 33e9,
    "spm_unit_capped_work_childcare_expenses": 348e9,
    "spm_unit_capped_housing_subsidy": 35e9,
    # Income transfers
    "tanf": 7_788_317_474.55,
    "alimony_income": 13e9,
    "alimony_expense": 13e9,
    # SS splits
    "social_security_retirement": 1_060e9,
    "social_security_disability": 148e9,
    "social_security_survivors": 160e9,
    "social_security_dependents": 84e9,
    # Retirement contributions
    "traditional_ira_contributions": 13.771289e9,
    "traditional_401k_contributions": 482.7e9,
    "roth_401k_contributions": 85.2e9,
    "roth_ira_contributions": 34.951077e9,
    "self_employed_pension_contribution_ald": 30.130848e9,
    # Wealth
    "net_worth": 160e12,
    # Medicaid / ACA national (loss.py: MEDICAID_*, ACA_*; DB: same)
    "medicaid": 871.7e9,
    "medicaid_enrollment_count": 72_429_055,
    "aca_ptc_spending": 98e9,
    "aca_ptc_enrollment_count": 19_743_689,
    # National EITC overall (Treasury tax_expenditures.eitc; DB national row)
    "eitc_treasury": 67.33e9,
}

FALLBACK_CBO = {
    "snap": 93.7e9,
    "social_security": 1_200e9,
    "ssi": 60e9,
    "income_tax": 4_000e9,
    "unemployment_compensation": 34.66e9,
}

DATASET_CONFIGS = {
    "a": {"path": PATH_A, "label": LABEL_A},
    "b": {"path": PATH_B, "label": LABEL_B},
}

POINT_CHECKS = [
    # ── EXISTING (preserved verbatim) ─────────────────────────────
    ("SNAP", "snap", "cbo:snap", 0.20),
    ("Social Security", "social_security", "cbo:social_security", 0.20),
    ("SSI", "ssi", "cbo:ssi", 0.20),
    ("Employment income", "employment_income", 10e12, 0.30),
    ("AGI", "adjusted_gross_income", 15e12, 0.30),
    ("Income tax", "income_tax", "cbo:income_tax", 0.20),
    ("EITC", "eitc", 60e9, 0.30),
    ("Refundable CTC", "refundable_ctc", 120e9, 0.30),
    ("Tip income", "tip_income", HARD_CODED_TOTALS["tip_income"], 0.30),
    ("Real estate taxes", "real_estate_taxes", HARD_CODED_TOTALS["real_estate_taxes"], 0.30),
    ("Rent", "rent", HARD_CODED_TOTALS["rent"], 0.30),
    ("Person count", "__person_count__", 335e6, 0.30),
    ("Household count", "__household_count__", 130e6, 0.30),
    ("SSN 'NONE' count", "__ssn_none__", 13e6, 0.20),

    # ── EXPANDED: national targets BOTH datasets calibrate to ─────
    # Each row below is a target present in both loss.py
    # (enhanced_cps) and policy_data.db filtered by
    # target_config.yaml at period=2024, geo_level='national'.
    # JCT tax-expenditure targets (salt / charitable / medical /
    # QBI / mortgage interest) are intentionally omitted — they
    # would require running 5 reform Microsimulations per dataset.

    # CBO national programs
    ("Unemployment comp", "unemployment_compensation",
        "cbo:unemployment_compensation", 0.30),

    # Social Security benefit-type splits
    ("SS retirement", "social_security_retirement",
        HARD_CODED_TOTALS["social_security_retirement"], 0.20),
    ("SS disability", "social_security_disability",
        HARD_CODED_TOTALS["social_security_disability"], 0.30),
    ("SS survivors", "social_security_survivors",
        HARD_CODED_TOTALS["social_security_survivors"], 0.30),
    ("SS dependents", "social_security_dependents",
        HARD_CODED_TOTALS["social_security_dependents"], 0.30),

    # Medicaid (national)
    ("Medicaid spending", "medicaid",
        HARD_CODED_TOTALS["medicaid"], 0.20),
    ("Medicaid enrollment", "__medicaid_enrollment__",
        HARD_CODED_TOTALS["medicaid_enrollment_count"], 0.20),

    # ACA PTC (national)
    ("ACA PTC spending", "aca_ptc",
        HARD_CODED_TOTALS["aca_ptc_spending"], 0.30),
    ("ACA PTC enrollment", "__aca_ptc_enrollment__",
        HARD_CODED_TOTALS["aca_ptc_enrollment_count"], 0.30),

    # National EITC overall (Treasury value; complements legacy
    # "EITC" row above which uses the SOI ~$60B figure)
    ("EITC (Treasury)", "eitc",
        HARD_CODED_TOTALS["eitc_treasury"], 0.30),

    # Healthcare totals
    ("HIP w/o Part B", "health_insurance_premiums_without_medicare_part_b",
        HARD_CODED_TOTALS["health_insurance_premiums_without_medicare_part_b"], 0.30),
    ("Other med expenses", "other_medical_expenses",
        HARD_CODED_TOTALS["other_medical_expenses"], 0.30),
    ("Medicare Part B prem", "medicare_part_b_premium",
        HARD_CODED_TOTALS["medicare_part_b_premium"], 0.30),
    ("OTC health expenses", "over_the_counter_health_expenses",
        HARD_CODED_TOTALS["over_the_counter_health_expenses"], 0.30),

    # Family / SPM
    ("Child support expense", "child_support_expense",
        HARD_CODED_TOTALS["child_support_expense"], 0.30),
    ("Child support recv", "child_support_received",
        HARD_CODED_TOTALS["child_support_received"], 0.30),
    ("Childcare expenses", "spm_unit_capped_work_childcare_expenses",
        HARD_CODED_TOTALS["spm_unit_capped_work_childcare_expenses"], 0.30),
    ("Housing subsidy", "spm_unit_capped_housing_subsidy",
        HARD_CODED_TOTALS["spm_unit_capped_housing_subsidy"], 0.30),

    # Income transfers
    ("TANF", "tanf",
        HARD_CODED_TOTALS["tanf"], 0.30),
    ("Alimony income", "alimony_income",
        HARD_CODED_TOTALS["alimony_income"], 0.30),
    ("Alimony expense", "alimony_expense",
        HARD_CODED_TOTALS["alimony_expense"], 0.30),

    # Retirement contributions
    ("Trad IRA contribs", "traditional_ira_contributions",
        HARD_CODED_TOTALS["traditional_ira_contributions"], 0.30),
    ("Trad 401k contribs", "traditional_401k_contributions",
        HARD_CODED_TOTALS["traditional_401k_contributions"], 0.30),
    ("Roth 401k contribs", "roth_401k_contributions",
        HARD_CODED_TOTALS["roth_401k_contributions"], 0.30),
    ("Roth IRA contribs", "roth_ira_contributions",
        HARD_CODED_TOTALS["roth_ira_contributions"], 0.30),
    ("SE pension ALD", "self_employed_pension_contribution_ald",
        HARD_CODED_TOTALS["self_employed_pension_contribution_ald"], 0.30),

    # Wealth
    ("Net worth", "net_worth",
        HARD_CODED_TOTALS["net_worth"], 0.30),
]

# Targets calibrated by only one dataset. Printed at the end of
# step3 for transparency. Curated from loss.py and target_config.yaml
# (relative to policy_data.db at period=2024, geo_level='national').
ONLY_IN_LOSS_PY = [
    "SOI AGI grid × filing status (~528 cells: AGI / count / "
    "employment_income / business_net_profits / capital_gains_gross / "
    "ordinary_dividends / partnership_s_corp / qualified_dividends / "
    "taxable_interest / total_pension / total_social_security)",
    "SOI all-AGI taxable aggregates (~14 vars: business / capital / "
    "estate losses, exempt_interest, ira_distributions, "
    "rent_and_royalty, mortgage_interest_deductions, taxable_pension / "
    "SS, unemployment)",
    "Healthcare × age-decade × 4 expense types (~36 cells)",
    "AGI × SPM-threshold decile (count + dollars, ~20 cells)",
    "Population by single year of age (national, 0–85)",
    "Population by state + population under 5 by state",
    "Negative household market income (total + count)",
    "Infants count (national)",
    "spm_unit_spm_threshold total",
    "EITC × AGI × qualifying-children grid (Pub 1304 Table 2.5)",
    "EITC by state (returns + amount per state)",
    "JCT tax-expenditure targets — omitted from the report card; "
    "would require 5 reform sims per dataset",
]

ONLY_IN_DB_YAML = [
    "All district-level targets (~8,284 rows: person_count×age, "
    "household_count×snap, AGI, real_estate_taxes, "
    "total_self_employment_income, taxable_pension_income, eitc, "
    "refundable_ctc, non_refundable_ctc, unemployment_compensation, "
    "aca_ptc, tax_unit_count×aca_ptc)",
    "State-level TANF (amount + spm_unit_count)",
    "State-level CTC (refundable + non-refundable)",
    "State-level income_tax (IRS SOI scalar by state)",
    "State-level ACA marketplace counts (used_aca_ptc, "
    "selected_marketplace_plan_benchmark_ratio)",
    "Net capital gains national",
    "Tax unit partnership / S-corp income national",
    "Total self-employment income national + return count",
    "Medical expense deduction × tax_unit_itemizes (SOI)",
    "Qualified business income deduction national (SOI)",
    "tax_unit_count × eitc_child_count (national, by # qualifying kids)",
    "household_count × spm_unit_energy_subsidy_reported (LIHEAP)",
    "Person-count × age cohort buckets (national, 18 cohorts)",
]

RANGE_CHECKS = [
    ("Income tax floor", "income_tax", 1e12, None),
    ("Empl income floor", "employment_income", 5e12, None),
    ("Self-empl floor", "self_employment_income", 50e9, None),
    ("HH weight range", "__household_count__", 100e6, 200e6),
    ("Person weight range", "__person_count__", 250e6, 400e6),
    ("Poverty rate", "__poverty_rate__", 0.05, 0.30),
    ("Mean empl income", "__mean_empl_income__", 15_000, 80_000),
    ("Tips floor", "tip_income", 40e9, None),
    ("Mortgage interest > $1", "deductible_mortgage_interest", 1, None),
    ("Citizen pct", "__citizen_pct__", 0.80, 0.95),
]

CONSISTENCY_CHECKS = [
    ("Undocumented == SSN NONE", "__undocumented_matches_ssn_none__"),
]

STATE_CHECKS = {
    "aca": {
        "var": "aca_ptc",
        "map_to": "household",
        "period": 2025,
        "csv_path": CAL_DIR / "aca_spending_and_enrollment_2024.csv",
        "target_col": "spending",
        "tolerance": 0.70,
        "name": "ACA PTC by state",
    },
    "med": {
        "var": "medicaid_enrolled",
        "map_to": "household",
        "period": 2025,
        "csv_path": CAL_DIR / "medicaid_enrollment_2024.csv",
        "target_col": "enrollment",
        "tolerance": 0.45,
        "name": "Medicaid enrollment",
    },
}

# ── Compute helpers ───────────────────────────────────────────

def compute_value(sim, var_key, period=2024):
    try:
        if var_key == "__person_count__":
            return float(sim.calculate("person_weight", period=period).values.sum())
        if var_key == "__household_count__":
            return float(sim.calculate("household_weight", period=period).values.sum())
        if var_key == "__ssn_none__":
            mask = sim.calculate("ssn_card_type", period=period) == "NONE"
            return float(mask.sum())
        if var_key == "__medicaid_enrollment__":
            return float(
                (sim.calculate("medicaid", map_to="person", period=period) > 0).sum()
            )
        if var_key == "__aca_ptc_enrollment__":
            return float(
                (sim.calculate("aca_ptc", map_to="person", period=period) > 0).sum()
            )
        if var_key == "__poverty_rate__":
            # Avoid expanding SPM-unit poverty status out to every person, which
            # can spike memory on large national builds.
            in_poverty = sim.calculate("in_poverty", period=period).values.astype(float)
            spm_unit_weight = sim.calculate("spm_unit_weight", period=period).values
            spm_unit_size = sim.calculate("spm_unit_size", period=period).values
            person_weights = spm_unit_weight * spm_unit_size
            return float(np.average(in_poverty, weights=person_weights))
        if var_key == "__mean_empl_income__":
            return float(sim.calculate("employment_income", map_to="person", period=period).mean())
        if var_key == "__citizen_pct__":
            status = sim.calculate("immigration_status", period=period)
            weighted_counts = status.weights.groupby(status).sum()
            return float(weighted_counts["CITIZEN"] / weighted_counts.sum())
        return float(sim.calculate(var_key, period=period).sum())
    except Exception as e:
        return f"N/A ({e.__class__.__name__})"


def get_cbo_targets(sim, period=2024):
    targets = {}
    for key in [
        "snap",
        "social_security",
        "ssi",
        "income_tax",
        "unemployment_compensation",
    ]:
        targets[key] = resolve_cbo_target(sim, key, period)
    return targets


def resolve_cbo_target(sim, key, period=2024):
    try:
        val = sim.tax_benefit_system.parameters(period).calibration.gov.cbo._children[key]
        return float(val)
    except Exception:
        return FALLBACK_CBO[key]


def resolve_point_target(sim, target_src, period=2024):
    if isinstance(target_src, str) and target_src.startswith("cbo:"):
        return resolve_cbo_target(sim, target_src[4:], period)
    return target_src


def run_point_checks(sim, cbo_targets, period=2024):
    results = []
    for name, var, target_src, tol in POINT_CHECKS:
        target = cbo_targets[target_src[4:]] if isinstance(target_src, str) and target_src.startswith("cbo:") else target_src
        val = compute_value(sim, var, period)
        if isinstance(val, str):
            results.append({"name": name, "value": val, "target": target, "tol": tol, "pct_error": None})
        else:
            pct_err = abs(val - target) / target if target != 0 else 0
            results.append({"name": name, "value": val, "target": target, "tol": tol, "pct_error": pct_err})
    return results


def run_range_checks(sim, period=2024):
    results = []
    for name, var, lo, hi in RANGE_CHECKS:
        val = compute_value(sim, var, period)
        if isinstance(val, str):
            results.append({"name": name, "value": val, "lo": lo, "hi": hi, "passed": None})
        else:
            ok = (lo is None or val >= lo) and (hi is None or val <= hi)
            results.append({"name": name, "value": val, "lo": lo, "hi": hi, "passed": ok})
    return results


def run_state_check(sim, var, map_to, period, csv_path, target_col, tolerance, name):
    if not csv_path.exists():
        return {"name": name, "error": f"CSV not found: {csv_path}", "passed": None}
    targets = pd.read_csv(csv_path)

    if name == "ACA PTC by state":
        targets["spending"] = targets["spending"] * 12
        targets["spending"] = targets["spending"] * (98e9 / targets["spending"].sum())
        target_col = "spending"

    try:
        state_code_hh = sim.calculate("state_code", map_to="household").values
        values = sim.calculate(var, map_to=map_to, period=period)
    except Exception as e:
        return {"name": name, "error": str(e), "passed": None}

    errors, rows = [], []
    for _, row in targets.iterrows():
        state = row["state"]
        target_val = row[target_col]
        simulated = float(values[state_code_hh == state].sum())
        pct_err = abs(simulated - target_val) / target_val if target_val else 0
        errors.append(pct_err)
        rows.append([state, simulated, target_val, pct_err])

    over = sum(1 for e in errors if e > tolerance)
    worst_idx = int(np.argmax(errors))
    return {
        "name": name,
        "passed": over == 0,
        "median_error": float(np.median(errors)),
        "n_over_tol": over,
        "n_states": len(rows),
        "tolerance": tolerance,
        "worst_state": rows[worst_idx][0],
        "worst_error": rows[worst_idx][3],
        "rows": rows,
    }


def run_consistency_check(sim, check_key, name, period=2024):
    try:
        if check_key == "__undocumented_matches_ssn_none__":
            ssn_type_none = sim.calculate("ssn_card_type", period=period) == "NONE"
            undocumented = sim.calculate("immigration_status", period=period) == "UNDOCUMENTED"
            mismatches = int((ssn_type_none != undocumented).sum())
            return {
                "name": name,
                "value": mismatches,
                "detail": f"{mismatches} mismatches",
                "passed": mismatches == 0,
            }
    except Exception as e:
        return {
            "name": name,
            "value": f"N/A ({e.__class__.__name__})",
            "detail": str(e),
            "passed": None,
        }

    return {
        "name": name,
        "value": "N/A (Unknown check)",
        "detail": f"Unknown check key: {check_key}",
        "passed": None,
    }


def get_dataset_config(dataset_key):
    return DATASET_CONFIGS[dataset_key]


# ── Formatting helpers ────────────────────────────────────────

def green(s):
    return f"\033[92m{s}\033[0m" if USE_COLOR else str(s)

def red(s):
    return f"\033[91m{s}\033[0m" if USE_COLOR else str(s)

def bold(s):
    return f"\033[1m{s}\033[0m" if USE_COLOR else str(s)

def fmt(x):
    ax = abs(x)
    if ax >= 1e12:
        return f"${x / 1e12:.2f}T"
    if ax >= 1e9:
        return f"${x / 1e9:.1f}B"
    if ax >= 1e6:
        return f"{x / 1e6:.1f}M"
    if ax >= 1e3:
        return f"${x / 1e3:.1f}k"
    return f"{x:.2f}"

def hr(char="─", width=100):
    print(char * width)
