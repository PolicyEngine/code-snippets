"""Read results JSON files and print the report.

Pass --text to also write a plain-text copy (no ANSI codes) to report.txt.
"""

import json
import re
import sys
from common import LABEL_A, LABEL_B, VERBOSE, green, red, bold, fmt, hr

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
        status_a = (green("PASS") if ra["passed"] else red("FAIL")) if ra["passed"] is not None else "N/A"
        val_b_str = fmt(rb["value"]) if rb["passed"] is not None else str(rb["value"])[:14]
        status_b = (green("PASS") if rb["passed"] else red("FAIL")) if rb["passed"] is not None else "N/A"

        print(f"{ra['name']:<24s} {range_str:>20s}  {val_a_str:>14s}  {val_b_str:>14s}  {status_a:>6s}  {status_b:>6s}")
    hr()


def print_state_report(result_a, result_b):
    print()
    name = result_a["name"]
    print(bold(f"STATE CHECK: {name}"))
    hr()

    for label, res in [(LABEL_A, result_a), (LABEL_B, result_b)]:
        if "error" in res:
            print(f"  {label}: {red(res['error'])}")
        else:
            print(f"  {label}: median err {res['median_error']:.1%}, "
                  f"{res['n_over_tol']} states over {res['tolerance']:.0%}, "
                  f"worst: {res['worst_state']} ({res['worst_error']:.1%})")

    if "error" not in result_a and "error" not in result_b:
        if result_a["median_error"] < result_b["median_error"]:
            print(f"  Winner (median error): {green(LABEL_A)}")
        elif result_b["median_error"] < result_a["median_error"]:
            print(f"  Winner (median error): {green(LABEL_B)}")
        else:
            print(f"  Winner (median error): tie")

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


def print_summary(pt_a, pt_b, aca_a, aca_b, med_a, med_b):
    print()
    print(bold("SUMMARY"))
    hr()
    a_wins = b_wins = 0
    for ra, rb in zip(pt_a, pt_b):
        if ra["pct_error"] is not None and rb["pct_error"] is not None:
            if ra["pct_error"] < rb["pct_error"]:
                a_wins += 1
            elif rb["pct_error"] < ra["pct_error"]:
                b_wins += 1
    for res_a, res_b in [(aca_a, aca_b), (med_a, med_b)]:
        if "median_error" in res_a and "median_error" in res_b:
            if res_a["median_error"] < res_b["median_error"]:
                a_wins += 1
            elif res_b["median_error"] < res_a["median_error"]:
                b_wins += 1
    print(f"  {LABEL_A} wins: {a_wins}")
    print(f"  {LABEL_B} wins: {b_wins}")
    hr()


# ── Main ──────────────────────────────────────────────────────

with open("results_a.json") as f:
    a = json.load(f)
with open("results_b.json") as f:
    b = json.load(f)

print_point_report(a["pt"], b["pt"])
if a["rng"] and b["rng"]:
    print_range_report(a["rng"], b["rng"])
if a.get("aca") and b.get("aca") and "name" in a["aca"] and "name" in b["aca"]:
    print_state_report(a["aca"], b["aca"])
if a.get("med") and b.get("med") and "name" in a["med"] and "name" in b["med"]:
    print_state_report(a["med"], b["med"])
print_summary(a["pt"], b["pt"], a.get("aca", {}), b.get("aca", {}), a.get("med", {}), b.get("med", {}))

if WRITE_TEXT:
    ansi_re = re.compile(r"\033\[[0-9;]*m")
    with open("report.txt", "w") as f:
        for line in _lines:
            f.write(ansi_re.sub("", line))
    _orig_print(f"\nWrote report.txt")
