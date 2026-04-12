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

HARD_CODED_TOTALS = {
    "tip_income": 38e9 * 1.4,
    "real_estate_taxes": 500e9,
    "rent": 735e9,
}

FALLBACK_CBO = {
    "snap": 93.7e9,
    "social_security": 1_200e9,
    "ssi": 60e9,
    "income_tax": 4_000e9,
}

DATASET_CONFIGS = {
    "a": {"path": PATH_A, "label": LABEL_A},
    "b": {"path": PATH_B, "label": LABEL_B},
}

POINT_CHECKS = [
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
    for key in ["snap", "social_security", "ssi", "income_tax"]:
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
