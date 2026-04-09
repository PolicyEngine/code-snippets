import sys
import numpy as np
import h5py
from pathlib import Path
from policyengine_us import Microsimulation
from policyengine_core.enums import Enum

STATE_TARGETS = {
    "AK": 27_464, "AL": 386_195, "AR": 156_607, "AZ": 348_055,
    "CA": 1_784_653, "CO": 237_106, "CT": 129_000, "DC": 14_799,
    "DE": 44_842, "FL": 4_211_902, "GA": 1_305_114, "HI": 22_170,
    "IA": 111_423, "ID": 103_783, "IL": 398_814, "IN": 295_772,
    "KS": 171_376, "KY": 75_317, "LA": 212_493, "MA": 311_199,
    "MD": 213_895, "ME": 62_586, "MI": 418_100, "MN": 135_001,
    "MO": 359_369, "MS": 286_410, "MT": 66_336, "NC": 1_027_930,
    "ND": 38_535, "NE": 117_882, "NH": 65_117, "NJ": 397_942,
    "NM": 56_472, "NV": 99_312, "NY": 288_681, "OH": 477_793,
    "OK": 277_436, "OR": 145_509, "PA": 434_571, "RI": 36_121,
    "SC": 571_175, "SD": 52_974, "TN": 555_103, "TX": 3_484_632,
    "UT": 366_939, "VA": 400_058, "VT": 30_027, "WA": 272_494,
    "WI": 266_327, "WV": 51_046, "WY": 42_293,
}

SEED = 42
MAX_ITERS = 20


def write_h5(sim, source_path, output_path):
    with h5py.File(source_path, "r") as src:
        source_periods = {
            var: set(src[var].keys()) for var in src.keys()
        }

    data = {}
    for variable in sim.tax_benefit_system.variables:
        var_meta = sim.tax_benefit_system.variables[variable]
        allowed = source_periods.get(variable)
        if allowed is None:
            continue
        holder_periods = sim.get_holder(variable).get_known_periods()
        if not holder_periods:
            continue
        data[variable] = {}
        for time_period in holder_periods:
            if str(time_period) not in allowed:
                continue
            values = sim.get_holder(variable).get_array(time_period)
            if var_meta.value_type in (Enum, str):
                if hasattr(values, "decode_to_str"):
                    values = values.decode_to_str().astype("S")
                else:
                    values = values.astype("S")
            else:
                values = np.array(values)
            if values is not None:
                data[variable][time_period] = values

    holder = sim.get_holder("person_id")
    if holder.get_known_periods():
        allowed = source_periods.get("person_id")
        if allowed:
            data["person_id"] = {}
            for tp in holder.get_known_periods():
                if str(tp) in allowed:
                    data["person_id"][tp] = np.array(holder.get_array(tp))

    with h5py.File(output_path, "w") as f:
        for variable, periods in data.items():
            grp = f.create_group(variable)
            for period, values in periods.items():
                grp.create_dataset(str(period), data=values)


def count_aca_persons(dataset_path, new_takeup):
    sim = Microsimulation(dataset=dataset_path)
    sim.set_input("takes_up_aca_if_eligible", 2024, new_takeup)
    has_ptc = sim.calculate("aca_ptc", period=2025, map_to="person") > 0
    eligible = sim.calculate("is_aca_ptc_eligible", period=2025)
    return float((has_ptc * eligible).sum()), sim


def process_state(state, base_dir, output_dir):
    target = STATE_TARGETS[state]
    tolerance = max(500, int(target * 0.02))
    dataset_path = str(base_dir / f"{state}.h5")

    output_path = output_dir / f"{state}.h5"
    if output_path.exists():
        print(f"\n{state}: SKIP (output already exists)")
        return True

    print(f"\n{'='*60}")
    print(f"{state}: target={target:,}, tolerance={tolerance:,}")

    sim0 = Microsimulation(dataset=dataset_path)
    takeup = sim0.calculate("takes_up_aca_if_eligible", period=2025).values
    weight = sim0.calculate("tax_unit_weight", period=2025).values
    total_weight = weight.sum()
    current_true_weight = weight[takeup == 1].sum()

    has_ptc_base = sim0.calculate("aca_ptc", period=2025, map_to="person") > 0
    eligible_base = sim0.calculate("is_aca_ptc_eligible", period=2025)
    baseline_persons = float((has_ptc_base * eligible_base).sum())
    print(f"  baseline persons: {baseline_persons:,.0f}")
    if baseline_persons >= target:
        print(f"  SKIP: baseline ({baseline_persons:,.0f}) already >= target ({target:,})")
        return False

    false_indices = np.where(takeup == 0)[0]
    rng = np.random.default_rng(SEED)
    rng.shuffle(false_indices)
    cum_weights = np.cumsum(weight[false_indices])

    def flip_takeup(additional_weight_needed):
        if additional_weight_needed <= 0:
            return takeup.copy().astype(bool)
        n_flips = np.searchsorted(cum_weights, additional_weight_needed, side="left") + 1
        n_flips = min(n_flips, len(false_indices))
        new_takeup = takeup.copy().astype(bool)
        new_takeup[false_indices[:n_flips]] = True
        return new_takeup

    lo, hi = 0.0, 1.0 - (current_true_weight / total_weight)
    best_sim = None
    best_takeup = None
    best_persons = None

    for iteration in range(MAX_ITERS):
        mid = (lo + hi) / 2
        additional = mid * total_weight
        new_takeup = flip_takeup(additional)
        rate = np.average(new_takeup, weights=weight)
        persons, sim = count_aca_persons(dataset_path, new_takeup)
        print(f"  iter {iteration}: rate={rate:.6f}, persons={persons:,.0f}")

        if abs(persons - target) < tolerance:
            best_sim = sim
            best_takeup = new_takeup
            best_persons = persons
            break
        elif rate >= 0.9999 and persons < target:
            print(f"  PLATEAU: rate ~1.0 but persons={persons:,.0f} < target={target:,}")
            best_sim = sim
            best_takeup = new_takeup
            best_persons = persons
            break
        elif persons < target:
            lo = mid
        else:
            hi = mid
            best_sim = sim
            best_takeup = new_takeup
            best_persons = persons

    if best_sim is None:
        print(f"  WARNING: {state} did not converge after {MAX_ITERS} iterations")
        return False

    output_path = output_dir / f"{state}.h5"
    write_h5(best_sim, dataset_path, str(output_path))
    print(f"  Written to {output_path} (persons: {best_persons:,.0f}, target: {target:,})")
    return True


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python hack_aca_takeup_states.py <base_dir>")
        print("  e.g. python hack_aca_takeup_states.py /home/baogorek/devl/huggingface/test/mar")
        sys.exit(1)

    base_dir = Path(sys.argv[1])
    output_dir = base_dir / "aca"
    output_dir.mkdir(exist_ok=True)

    h5_files = sorted(base_dir.glob("*.h5"))
    states = [f.stem for f in h5_files if f.stem in STATE_TARGETS]
    print(f"Found {len(states)} state files in {base_dir}")
    print(f"Output directory: {output_dir}")

    succeeded, failed = [], []
    for state in states:
        try:
            ok = process_state(state, base_dir, output_dir)
            (succeeded if ok else failed).append(state)
        except Exception as e:
            print(f"  ERROR: {e}")
            failed.append(state)

    print(f"\n{'='*60}")
    print(f"Done: {len(succeeded)} succeeded, {len(failed)} failed")
    if failed:
        print(f"Failed: {', '.join(failed)}")
