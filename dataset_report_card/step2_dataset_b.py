"""Run all Dataset B checks in short-lived subprocesses and write results_b.json."""

import json
from pathlib import Path
import subprocess
import sys

from common import CONSISTENCY_CHECKS, FALLBACK_CBO, LABEL_B, POINT_CHECKS, RANGE_CHECKS, STATE_CHECKS

DATASET_KEY = "b"


def point_target_value(target_src):
    if isinstance(target_src, str) and target_src.startswith("cbo:"):
        return FALLBACK_CBO[target_src[4:]]
    return target_src


def failure_result(kind, identifier, error):
    if kind == "pt":
        name, _, target_src, tol = POINT_CHECKS[identifier]
        return {
            "name": name,
            "value": f"ERROR ({error})",
            "target": point_target_value(target_src),
            "tol": tol,
            "pct_error": None,
            "error": error,
        }
    if kind == "rng":
        name, _, lo, hi = RANGE_CHECKS[identifier]
        return {
            "name": name,
            "value": f"ERROR ({error})",
            "lo": lo,
            "hi": hi,
            "passed": None,
            "error": error,
        }
    if kind == "cons":
        name, _ = CONSISTENCY_CHECKS[identifier]
        return {
            "name": name,
            "value": f"ERROR ({error})",
            "detail": error,
            "passed": None,
            "error": error,
        }
    return {
        "name": STATE_CHECKS[kind]["name"],
        "error": error,
        "passed": None,
    }


def run_indexed_checks(kind, count):
    results = []
    for i in range(count):
        result_path = Path(f"results_{DATASET_KEY}_{kind}_{i}.json")
        result_path.unlink(missing_ok=True)
        try:
            subprocess.run([sys.executable, "step_part.py", DATASET_KEY, kind, str(i)], check=True)
            with open(result_path) as f:
                results.append(json.load(f))
        except subprocess.CalledProcessError as exc:
            error = f"subprocess failed with return code {exc.returncode}"
            print(f"{LABEL_B}: {kind}[{i}] failed: {error}")
            result = failure_result(kind, i, error)
            with open(result_path, "w") as f:
                json.dump(result, f)
            results.append(result)
    return results


def run_state_check(kind):
    result_path = Path(f"results_{DATASET_KEY}_{kind}.json")
    result_path.unlink(missing_ok=True)
    try:
        subprocess.run([sys.executable, "step_part.py", DATASET_KEY, kind], check=True)
        with open(result_path) as f:
            return json.load(f)
    except subprocess.CalledProcessError as exc:
        error = f"subprocess failed with return code {exc.returncode}"
        print(f"{LABEL_B}: {kind} failed: {error}")
        result = failure_result(kind, kind, error)
        with open(result_path, "w") as f:
            json.dump(result, f)
        return result


print(f"Loading {LABEL_B} through subprocess checks...")

pt = run_indexed_checks("pt", len(POINT_CHECKS))
rng = run_indexed_checks("rng", len(RANGE_CHECKS))
consistency = run_indexed_checks("cons", len(CONSISTENCY_CHECKS))
aca = run_state_check("aca")
med = run_state_check("med")

with open("results_b.json", "w") as f:
    json.dump(
        {
            "pt": pt,
            "rng": rng,
            "consistency": consistency,
            "aca": aca,
            "med": med,
        },
        f,
    )

print("Wrote results_b.json")
