"""Shared config, helpers, and check definitions for the report card."""

import os
from pathlib import Path
import numpy as np
import pandas as pd

# Workaround for policyengine_core/enums/enum.py:84 — `array.astype(str)` ASCII-decodes byte arrays, fails on "DOÑA_ANA_COUNTY_NM" (county/2024) in the local US.h5 build.
def _patch_policyengine_enum_utf8():
    from policyengine_core.enums.enum import Enum as _PEEnum
    _orig = _PEEnum.encode
    @classmethod
    def encode(cls, array):
        if isinstance(array, np.ndarray) and array.dtype.kind == "S":
            array = np.char.decode(array, "utf-8")
        return _orig.__func__(cls, array)
    _PEEnum.encode = encode
_patch_policyengine_enum_utf8()

# ── Config ────────────────────────────────────────────────────
# Root of a local policyengine-us-data checkout. Override per machine via
# POLICYENGINE_US_DATA_ROOT; default assumes ~/policyengine-us-data.
_DATA_ROOT = Path(os.environ.get(
    "POLICYENGINE_US_DATA_ROOT",
    str(Path.home() / "policyengine-us-data"),
))

PATH_A = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
PATH_B = str(_DATA_ROOT / "local_area_build" / "national" / "US.h5")
LABEL_A = "enhanced_cps_2024.h5"
LABEL_B = "US.h5"
VERBOSE = False
USE_COLOR = True
INCLUDE_RANGE_CHECKS = os.environ.get("REPORT_SKIP_RANGE", "").lower() not in ("1", "true", "yes")

CAL_DIR = _DATA_ROOT / "policyengine_us_data" / "storage" / "calibration_targets"
CALIBRATION_LOG_PATH = _DATA_ROOT / "policyengine_us_data" / "storage" / "calibration" / "national" / "calibration_log.csv"
_calib_log_cache = None

HARD_CODED_TOTALS = {
    # source: Social Security tips (~$38B) uprated 40% for underreporting.
    # Matches policy_data.db target_id=16, notes="Social security tips uprated 40%
    # to account for underreporting | Source: IRS Form W-2 Box 7 statistics".
    "tip_income": 38e9 * 1.4,
    # source: PolicyEngine hard-coded national property tax target, Census Bureau
    # basis (policy_data.db target_id=14, source="PolicyEngine", notes="Property
    # taxes paid | Source: Census Bureau"; see etl_national_targets.py and
    # utils/loss.py which label this a rough estimate between $350B and $600B).
    # Matches the calibration's unfiltered national target exactly. US.h5's 12.4%
    # under-target is a legitimate build-side signal, not a drift.
    "real_estate_taxes": 500e9,
    # source: policy_data.db target_id=15, source="PolicyEngine", notes="Rental
    # payments | Source: Census Bureau/BLS".
    "rent": 735e9,
}

FALLBACK_CBO = {
    # Defense-in-depth only — used if the policy parameter lookup in
    # resolve_cbo_target fails. The primary source is policyengine_us's
    # parameters().calibration.gov.cbo._children, which is what lands in
    # policy_data.db as source="PolicyEngine", notes="CBO Budget Projections".
    # Canonical 2024 values (for reference):
    #   snap=$93.730B (policy_data.db target_id=26)
    #   social_security=$1.454T (target_id=27)
    #   ssi=$57.0B (target_id=28)
    #   income_tax_positive=$2.426T filer-filtered (target_id=30)
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
    # source: CBO Budget Projections, resolved via policy parameter tree
    # parameters().calibration.gov.cbo.snap (see resolve_cbo_target).
    # Matches policy_data.db target_id=26 ($93.730B 2024).
    ("SNAP", "snap", "cbo:snap", 0.20),
    # source: CBO Budget Projections via parameters().calibration.gov.cbo.social_security.
    # Matches policy_data.db target_id=27 ($1.454T 2024).
    ("Social Security", "social_security", "cbo:social_security", 0.20),
    # source: CBO Budget Projections via parameters().calibration.gov.cbo.ssi.
    # Matches policy_data.db target_id=28 ($57.0B 2024).
    ("SSI", "ssi", "cbo:ssi", 0.20),
    # source: TBD — no matching row in policy_data.db for employment_income at any
    # geo level. The $10T figure is undocumented in the codebase; git blame points
    # only at local commits. This should be sourced from BEA NIPA wages & salaries
    # or CBO projections in a follow-up.
    ("Employment income", "employment_income", 10e12, 0.30),
    # source: IRS SOI Pub 1304 Table 1.1 TY2023 AGI for filers (policy_data.db target_id=9509,
    # stratum tax_unit_is_filer==1), CPI-uprated to 2024 by the calibration pipeline.
    ("AGI (filer)", "__filer_agi__", 15.828848018564848e12, 0.30),
    # source: CBO Budget Projections via parameters().calibration.gov.cbo.income_tax.
    # Resolves at runtime to ~$2.43T, which matches policy_data.db target_id=30
    # (income_tax_positive, filer-filtered, $2.426T 2024).
    ("Income tax", "income_tax", "cbo:income_tax", 0.20),
    # source: Treasury/JCT Tax Expenditures (EITC), policy_data.db target_id=31
    # ($67.330B 2024, filer-filtered, source="PolicyEngine", notes="EITC tax
    # expenditure | Source: Treasury/JCT Tax Expenditures"). Previous value $60B
    # was stale; this is a drift fix.
    ("EITC", "eitc", 67.33e9, 0.30),
    # source: IRS SOI 2022 refundable CTC for filers with refundable_ctc>0
    # (policy_data.db target_id=9470, source="IRS SOI", notes="IRS geography-file
    # national aggregate target"), CPI-uprated to 2024 by the calibration pipeline.
    # Non-filer refundable_ctc is 0 in the PolicyEngine formula, so the unfiltered
    # sim.calculate().sum() is directly comparable to the filer-filtered target.
    ("Refundable CTC", "refundable_ctc", 36.438422319e9, 0.30),
    ("Tip income", "tip_income", HARD_CODED_TOTALS["tip_income"], 0.30),
    ("Real estate taxes", "real_estate_taxes", HARD_CODED_TOTALS["real_estate_taxes"], 0.30),
    ("Rent", "rent", HARD_CODED_TOTALS["rent"], 0.30),
    # source: TBD — no policy_data.db row for person_count. 335M is a rough US
    # population estimate; sourced from Census / BEA population would be better.
    ("Person count", "__person_count__", 335e6, 0.30),
    # source: TBD — policy_data.db has no *unfiltered* national household_count.
    # The only existing rows (target_id=39,40) are LIHEAP-filtered household
    # counts at ~$5.9M. 130M is undocumented; likely a rough ACS estimate.
    ("Household count", "__household_count__", 130e6, 0.30),
    # source: TBD — no policy_data.db row for ssn_card_type=="NONE". 13M is an
    # undocumented estimate of undocumented/non-SSA population.
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
        if var_key == "__filer_agi__":
            agi = sim.calculate("adjusted_gross_income", period=period).values
            is_filer = sim.calculate("tax_unit_is_filer", period=period).values.astype(bool)
            weights = sim.calculate("tax_unit_weight", period=period).values
            return float((agi[is_filer] * weights[is_filer]).sum())
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


def _load_calib_log():
    global _calib_log_cache
    if _calib_log_cache is not None:
        return _calib_log_cache
    if not CALIBRATION_LOG_PATH.exists():
        raise FileNotFoundError(
            f"Calibration log not found at {CALIBRATION_LOG_PATH}. "
            f"Set POLICYENGINE_US_DATA_ROOT or run the calibration step."
        )
    df = pd.read_csv(
        CALIBRATION_LOG_PATH,
        usecols=["target_name", "epoch", "target", "estimate"],
    )
    df["epoch"] = df["epoch"].astype(int)
    _calib_log_cache = df
    return df


def _lookup_calib_row(target_name, epoch):
    df = _load_calib_log()
    sub = df[(df["epoch"] == int(epoch)) & (df["target_name"] == target_name)]
    if sub.empty:
        var_segment = target_name.split("/")[1] if target_name.count("/") >= 2 else ""
        candidates = (
            df[df["target_name"].str.contains(f"/{var_segment}/", regex=False, na=False)]
            ["target_name"].unique().tolist()[:5]
        )
        raise ValueError(
            f"Calibration log has no entry for target {target_name!r} at epoch {epoch}. "
            f"Near-matches for variable {var_segment!r}: {candidates}"
        )
    return sub.iloc[0]


def resolve_calib_target(target_name, epoch=4000):
    return float(_lookup_calib_row(target_name, epoch)["target"])


def resolve_calib_estimate(target_name, epoch=4000):
    return float(_lookup_calib_row(target_name, epoch)["estimate"])


def resolve_point_target(sim, target_src, period=2024):
    if isinstance(target_src, str):
        if target_src.startswith("cbo:"):
            return resolve_cbo_target(sim, target_src[4:], period)
        if target_src.startswith("calib:"):
            return resolve_calib_target(target_src[6:])
    return target_src


def run_point_checks(sim, cbo_targets, period=2024):
    results = []
    for name, var, target_src, tol in POINT_CHECKS:
        target = resolve_point_target(sim, target_src, period)
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
