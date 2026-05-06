"""Read results JSON files and print the report.

Pass --text to also write a plain-text copy (no ANSI codes) to report.txt.
"""

import json
import re
import sys

from common import (
    LABEL_A,
    LABEL_B,
    ONLY_IN_DB_YAML,
    ONLY_IN_LOSS_PY,
    VERBOSE,
    bold,
    fmt,
    green,
    hr,
    red,
)

WRITE_TEXT = "--text" in sys.argv

_lines = []
_orig_print = print


def print(*args, **kwargs):
    _orig_print(*args, **kwargs)
    if WRITE_TEXT:
        import io

        buf = io.StringIO()
        _orig_print(*args, **{**kwargs, "file": buf})
        _lines.append(buf.getvalue())


def status_text(passed):
    if passed is True:
        return green("PASS")
    if passed is False:
        return red("FAIL")
    return "N/A"


def state_result_name(result_a, result_b):
    return result_a.get("name") or result_b.get("name") or "Unknown state check"


def normalize_state_result(result, name):
    if result and "name" in result:
        normalized = dict(result)
        if "passed" not in normalized and "n_over_tol" in normalized:
            normalized["passed"] = normalized["n_over_tol"] == 0
        if "n_states" not in normalized:
            normalized["n_states"] = len(normalized.get("rows", []))
        return normalized
    return {"name": name, "error": "Result missing", "passed": None}


def print_point_report(results_a, results_b):
    print()
    print(bold("POINT-TARGET CHECKS"))
    hr()
    hdr = f"{'Check':<22s} {'Target':>12s}  {LABEL_A:>14s}  {LABEL_B:>14s}  {'Err A':>7s}  {'Err B':>7s}  Winner"
    print(bold(hdr))
    hr()
    for ra, rb in zip(results_a, results_b):
        target_str = fmt(ra["target"])
        val_a_str = fmt(ra["value"]) if ra["pct_error"] is not None else str(ra["value"])[:14]
        err_a_str = f"{ra['pct_error']:.1%}" if ra["pct_error"] is not None else "N/A"
        val_b_str = fmt(rb["value"]) if rb["pct_error"] is not None else str(rb["value"])[:14]
        err_b_str = f"{rb['pct_error']:.1%}" if rb["pct_error"] is not None else "N/A"

        if ra["pct_error"] is None and rb["pct_error"] is None:
            winner = "—"
        elif ra["pct_error"] is None:
            winner = LABEL_B
        elif rb["pct_error"] is None:
            winner = LABEL_A
        elif ra["pct_error"] < rb["pct_error"]:
            winner = green(LABEL_A)
        elif rb["pct_error"] < ra["pct_error"]:
            winner = green(LABEL_B)
        else:
            winner = "tie"

        if ra["pct_error"] is not None:
            err_a_str = (green if ra["pct_error"] <= ra["tol"] else red)(err_a_str)
        if rb["pct_error"] is not None:
            err_b_str = (green if rb["pct_error"] <= rb["tol"] else red)(err_b_str)

        print(f"{ra['name']:<22s} {target_str:>12s}  {val_a_str:>14s}  {val_b_str:>14s}  {err_a_str:>7s}  {err_b_str:>7s}  {winner}")
    hr()


def print_range_report(results_a, results_b):
    print()
    print(bold("RANGE CHECKS"))
    hr()
    hdr = f"{'Check':<24s} {'Range':>20s}  {LABEL_A:>14s}  {LABEL_B:>14s}  {'A':>6s}  {'B':>6s}"
    print(bold(hdr))
    hr()
    for ra, rb in zip(results_a, results_b):
        lo, hi = ra["lo"], ra["hi"]
        if lo is not None and hi is not None:
            range_str = f"{fmt(lo)}–{fmt(hi)}"
        elif lo is not None:
            range_str = f"> {fmt(lo)}"
        else:
            range_str = f"< {fmt(hi)}"

        val_a_str = fmt(ra["value"]) if ra["passed"] is not None else str(ra["value"])[:14]
        val_b_str = fmt(rb["value"]) if rb["passed"] is not None else str(rb["value"])[:14]

        print(
            f"{ra['name']:<24s} {range_str:>20s}  {val_a_str:>14s}  {val_b_str:>14s}  "
            f"{status_text(ra['passed']):>6s}  {status_text(rb['passed']):>6s}"
        )
    hr()


def print_consistency_report(results_a, results_b):
    print()
    print(bold("CONSISTENCY CHECKS"))
    hr()
    hdr = f"{'Check':<28s} {LABEL_A:>24s}  {LABEL_B:>24s}  {'A':>6s}  {'B':>6s}"
    print(bold(hdr))
    hr()
    for ra, rb in zip(results_a, results_b):
        detail_a = ra.get("detail", str(ra.get("value", "N/A")))[:24]
        detail_b = rb.get("detail", str(rb.get("value", "N/A")))[:24]
        print(
            f"{ra['name']:<28s} {detail_a:>24s}  {detail_b:>24s}  "
            f"{status_text(ra.get('passed')):>6s}  {status_text(rb.get('passed')):>6s}"
        )
    hr()


def state_winner(result_a, result_b):
    if "error" in result_a and "error" in result_b:
        return "—"
    if "error" in result_a:
        return LABEL_B
    if "error" in result_b:
        return LABEL_A
    if result_a.get("passed") is True and result_b.get("passed") is False:
        return green(LABEL_A)
    if result_b.get("passed") is True and result_a.get("passed") is False:
        return green(LABEL_B)
    if result_a["median_error"] < result_b["median_error"]:
        return green(LABEL_A)
    if result_b["median_error"] < result_a["median_error"]:
        return green(LABEL_B)
    return "tie"


def print_state_report(result_a, result_b):
    name = state_result_name(result_a, result_b)
    result_a = normalize_state_result(result_a, name)
    result_b = normalize_state_result(result_b, name)

    print()
    print(bold(f"STATE CHECK: {name}"))
    hr()

    for label, res in [(LABEL_A, result_a), (LABEL_B, result_b)]:
        if "error" in res:
            print(f"  {label}: {red(res['error'])}")
        else:
            print(
                f"  {label}: {status_text(res.get('passed'))}, "
                f"median err {res['median_error']:.1%}, "
                f"{res['n_over_tol']}/{res['n_states']} states over {res['tolerance']:.0%}, "
                f"worst: {res['worst_state']} ({res['worst_error']:.1%})"
            )

    print(f"  Winner: {state_winner(result_a, result_b)}")

    if VERBOSE and "rows" in result_a:
        print()
        rows_a = {r[0]: r for r in result_a.get("rows", [])}
        rows_b = {r[0]: r for r in result_b.get("rows", [])}
        print(f"  {'State':<6s} {LABEL_A:>14s}  {LABEL_B:>14s}  {'Target':>14s}  {'Err A':>7s}  {'Err B':>7s}")
        for state in rows_a:
            ra = rows_a[state]
            rb = rows_b.get(state)
            rb_str = fmt(rb[1]) if rb else "N/A"
            rb_err = f"{rb[3]:.1%}" if rb else "N/A"
            print(f"  {state:<6s} {fmt(ra[1]):>14s}  {rb_str:>14s}  {fmt(ra[2]):>14s}  {ra[3]:.1%}  {rb_err:>7s}")
    hr()


def print_calibration_coverage_diff():
    """Print targets calibrated by only one of the two datasets.

    The point-target table above only includes targets present in
    BOTH calibration paths. This section lists what each side
    additionally calibrates against, so the comparison's blind
    spots are explicit. Lists are curated in common.py.
    """
    print()
    print(bold("CALIBRATION COVERAGE — TARGETS ONLY ONE SIDE CALIBRATES"))
    hr()
    print(bold(f"Only in {LABEL_A} (loss.py):"))
    for item in ONLY_IN_LOSS_PY:
        print(f"  - {item}")
    print()
    print(bold(f"Only in {LABEL_B} (target_config.yaml + policy_data.db):"))
    for item in ONLY_IN_DB_YAML:
        print(f"  - {item}")
    hr()


def print_summary(pt_a, pt_b, rng_a, rng_b, cons_a, cons_b, state_results):
    print()
    print(bold("SUMMARY"))
    hr()

    a_point_wins = b_point_wins = 0
    for ra, rb in zip(pt_a, pt_b):
        if ra["pct_error"] is not None and rb["pct_error"] is not None:
            if ra["pct_error"] < rb["pct_error"]:
                a_point_wins += 1
            elif rb["pct_error"] < ra["pct_error"]:
                b_point_wins += 1

    print(f"  Point-target wins: {LABEL_A} {a_point_wins}, {LABEL_B} {b_point_wins}")
    print(f"  Range passes: {LABEL_A} {sum(r['passed'] is True for r in rng_a)}/{len(rng_a)}, {LABEL_B} {sum(r['passed'] is True for r in rng_b)}/{len(rng_b)}")
    print(f"  Consistency passes: {LABEL_A} {sum(r.get('passed') is True for r in cons_a)}/{len(cons_a)}, {LABEL_B} {sum(r.get('passed') is True for r in cons_b)}/{len(cons_b)}")

    a_state_passes = sum(result.get("passed") is True for result in state_results["a"])
    b_state_passes = sum(result.get("passed") is True for result in state_results["b"])
    print(f"  State-check passes: {LABEL_A} {a_state_passes}/{len(state_results['a'])}, {LABEL_B} {b_state_passes}/{len(state_results['b'])}")
    hr()


with open("results_a.json") as f:
    a = json.load(f)
with open("results_b.json") as f:
    b = json.load(f)

pt_a = a.get("pt", [])
pt_b = b.get("pt", [])
rng_a = a.get("rng", [])
rng_b = b.get("rng", [])
cons_a = a.get("consistency", [])
cons_b = b.get("consistency", [])
aca_a = a.get("aca", {})
aca_b = b.get("aca", {})
med_a = a.get("med", {})
med_b = b.get("med", {})

aca_name = state_result_name(aca_a, aca_b)
med_name = state_result_name(med_a, med_b)
aca_a = normalize_state_result(aca_a, aca_name)
aca_b = normalize_state_result(aca_b, aca_name)
med_a = normalize_state_result(med_a, med_name)
med_b = normalize_state_result(med_b, med_name)

print_point_report(pt_a, pt_b)
if rng_a and rng_b:
    print_range_report(rng_a, rng_b)
if cons_a and cons_b:
    print_consistency_report(cons_a, cons_b)
if aca_a or aca_b:
    print_state_report(aca_a, aca_b)
if med_a or med_b:
    print_state_report(med_a, med_b)
print_summary(
    pt_a,
    pt_b,
    rng_a,
    rng_b,
    cons_a,
    cons_b,
    {"a": [aca_a, med_a], "b": [aca_b, med_b]},
)
print_calibration_coverage_diff()

if WRITE_TEXT:
    ansi_re = re.compile(r"\033\[[0-9;]*m")
    with open("report.txt", "w") as f:
        for line in _lines:
            f.write(ansi_re.sub("", line))
    _orig_print("\nWrote report.txt")
