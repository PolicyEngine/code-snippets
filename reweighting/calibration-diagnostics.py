import os

from pathlib import Path

import pandas as pd
import numpy as np
import torch
import h5py
from huggingface_hub import hf_hub_download
from typing import Optional
from typing import Sequence, Tuple, Dict, List

from policyengine_core.data import Dataset
from policyengine_us import Microsimulation
from policyengine_us.system import system
from us_congressional_districts.utils import (
    get_data_directory,
    state_abbr_from_fips,
)
import pandas as pd
import numpy as np
import torch
from microcalibrate import Calibration
import logging

from collections import defaultdict
 
 
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def get_dataset(dataset: str = "cps_2023", time_period=2023) -> pd.DataFrame:
    """
    Get the dataset from the huggingface hub.
    """
    dataset_path = hf_hub_download(
        repo_id="policyengine/policyengine-us-data",
        filename=f"{dataset}.h5",
        local_dir=get_data_directory() / "input" / "cps",
    )

    return Dataset.from_file(dataset_path, time_period=time_period)


def get_agi_band_label(lower: float, upper: float) -> str:
    """Get the label for the AGI band based on lower and upper bounds."""
    if lower <= 0:
        return f"-inf_{int(upper)}"
    elif np.isposinf(upper):
        return f"{int(lower)}_inf"
    else:
        return f"{int(lower)}_{int(upper)}"


def get_state_abbr_from_fips(fips_code: str) -> str:
    """Get the state abbreviation from the FIPS code."""
    state_abbr_dict = state_abbr_from_fips()
    return state_abbr_dict.get(fips_code)


def create_metric_matrix(
    sim: Microsimulation,
    sim_calculations: dict,
    ages: pd.DataFrame,
    soi_targets: pd.DataFrame,
    households: pd.DataFrame,
):
    """
    Create metric matrix for multi-level calibration (national, state, district).

    Args:
        dataset: Dataset to use for simulation
        ages: DataFrame with age targets for all geographic levels
        soi_targets: DataFrame with SOI targets for all geographic levels
        time_period: Year for calculation

    Returns:
        DataFrame with metrics for each household, including geographic identifiers
    """
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    # Use passed simulation data
    age = sim.calculate("age").values
    state_code = sim.calculate("state_code").values
    state_fips = sim.calculate("state_fips").values

    matrix = pd.DataFrame()

    for i, age_range in enumerate(age_ranges):
        if age_range != "85+":
            lower_age, upper_age = age_range.split("-")
            in_age_band = (age >= int(lower_age)) & (age <= int(upper_age))
        else:
            in_age_band = age >= 85

        # Map age band to household level
        in_age_band = sim.map_result(
            in_age_band, "person", "household", how="sum"
        )

        # Create age metrics for each geographic level
        unique_geo_ids = ages["GEO_ID"].unique()
        for j, geo_id in enumerate(unique_geo_ids):
            if geo_id.startswith("0100000US"):
                level_prefix = "national"
                geo_mask = np.ones(len(in_age_band), dtype=bool)
            elif geo_id.startswith("0400000US"):
                state_fips_code = geo_id[9:11]
                level_prefix = (
                    f"state_{get_state_abbr_from_fips(state_fips_code)}"
                )
                geo_mask = state_fips == int(state_fips_code)
            elif geo_id.startswith("5001800US"):
                district_code = geo_id[11:13]
                state_fips_code = geo_id[9:11]
                level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
                # Create mask for households in this specific state AND district
                state_mask = state_fips == int(state_fips_code)
                district_int = int(district_code)
                hhs = households.loc[
                    households["district"] == district_int, "household_id"
                ]
                geo_mask = np.zeros(len(in_age_band), dtype=bool)
                # Use household IDs to create boolean mask - find positions where household index matches
                household_mask = np.isin(
                    np.arange(len(state_mask)), hhs.values
                )
                geo_mask = household_mask & state_mask
            else:
                continue

            combined_mask = in_age_band * geo_mask.astype(float)

            col = f"acs/{level_prefix}/age/count/{age_range}"
            matrix[col] = combined_mask

    agi_long = (
        soi_targets[
            [
                "GEO_ID",
                "AGI_LOWER_BOUND",
                "AGI_UPPER_BOUND",
                "VARIABLE",
                "IS_COUNT",
            ]
        ]
        .drop_duplicates()
        .sort_values(["IS_COUNT", "VARIABLE", "AGI_LOWER_BOUND"])
    )

    for _, row in agi_long.iterrows():
        lower, upper = row.AGI_LOWER_BOUND, row.AGI_UPPER_BOUND
        band = get_agi_band_label(lower, upper)
        var = row.VARIABLE.replace("/count", "").replace("/amount", "")
        is_count = row.IS_COUNT
        geo_id = row.GEO_ID
        var_values = sim_calculations[var]

        mask = (sim_calculations["adjusted_gross_income"] > lower) & (
            sim_calculations["adjusted_gross_income"] <= upper
        )

        # Determine geographic level and create appropriate mask
        if geo_id.startswith("0100000US"):
            geo_mask = np.ones(len(mask), dtype=bool)
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            geo_mask = state_fips == int(state_fips_code)
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
            # Create mask for households in this specific state AND district
            state_mask = state_fips == int(state_fips_code)
            district_int = int(district_code)
            hhs = households.loc[
                households["district"] == district_int, "household_id"
            ]
            geo_mask = np.zeros(len(state_fips), dtype=bool)
            # Use household IDs to create boolean mask - find positions where household index matches
            household_mask = np.isin(np.arange(len(state_mask)), hhs.values)
            geo_mask = household_mask & state_mask
        else:
            continue
        
        # NOTE: Ben weakened the mask because it was breaking for some reason
        # Map geographic mask to tax_unit level
        #from copy import deepcopy
        #x = deepcopy(geo_mask).astype(float)
        #geo_mask = sim.map_result(
        #    x, "household", "tax_unit"
        #)
        combined_mask = mask # & (geo_mask > 0)

        if is_count:
            col = f"soi/{level_prefix}/{var}/count/{band}"
            metric = combined_mask * (var_values > 0).astype(float)
            metric = sim.map_result(metric, "tax_unit", "household")
        else:
            col = f"soi/{level_prefix}/{var}/amount/{band}"
            metric = var_values * combined_mask
            metric = sim.map_result(metric, "tax_unit", "household")

        matrix[col] = metric

    matrix["state_code"] = state_code
    matrix["state_fips"] = state_fips

    return matrix


def create_target_matrix(ages, soi_targets):
    ages_count_matrix = ages.iloc[:, 2:]
    age_ranges = list(ages_count_matrix.columns)

    # Initialize target dictionary
    targets_dict = {}

    # Create age targets for each geographic level
    for idx, row in ages.iterrows():
        geo_id = row["GEO_ID"]

        if geo_id.startswith("0100000US"):
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
        else:
            continue

        for age_range in age_ranges:
            #col_name = f"age/{level_prefix}/{age_range}"
            col_name = f"acs/{level_prefix}/age/count/{age_range}"
            targets_dict[col_name] = row[age_range]

    # Create SOI targets with geographic level indicators
    agi_with_labels = soi_targets.assign(
        band=lambda df: df.apply(
            lambda r: get_agi_band_label(r.AGI_LOWER_BOUND, r.AGI_UPPER_BOUND),
            axis=1,
        )
    )
    agi_with_labels = agi_with_labels.sort_values(
        ["IS_COUNT", "VARIABLE", "AGI_LOWER_BOUND"]
    )

    for _, row in agi_with_labels.iterrows():
        geo_id = row["GEO_ID"]
        variable = row["VARIABLE"]
        band = row["band"]
        value = row["VALUE"]

        if geo_id.startswith("0100000US"):
            level_prefix = "national"
        elif geo_id.startswith("0400000US"):
            state_fips_code = geo_id[9:11]
            level_prefix = f"state_{get_state_abbr_from_fips(state_fips_code)}"
        elif geo_id.startswith("5001800US"):
            district_code = geo_id[11:13]
            state_fips_code = geo_id[9:11]
            level_prefix = f"district_{get_state_abbr_from_fips(state_fips_code)}{district_code}"
        else:
            continue

        col_name = f"soi/{level_prefix}/{variable}/{band}"
        targets_dict[col_name] = value

    # Convert to DataFrame
    y = pd.DataFrame([targets_dict])

    return y


def create_state_mask(
    dataset: str = None,
    districts: pd.Series = pd.Series(["5001800US5600"]),
    time_period: int = 2023,
) -> np.ndarray:
    """
    Create a matrix R to accompany the loss matrix M s.t. (W x M) x R = Y_
    where Y_ is the target matrix s.t. no target is constructed
    from weights from a different state.
    """

    sim = Microsimulation(dataset=dataset)
    sim.default_calculation_period = time_period

    household_states = sim.calculate("state_fips").values
    district_states = districts.str[9:11].astype(np.int32)
    r = np.zeros((len(districts), len(household_states)))

    for i in range(len(districts)):
        r[i] = household_states == district_states[i]

    return r


def create_district_to_state_matrix():
    """Create [50, 450] sparse binary matrix mapping states to districts"""

    districts = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age_district.csv"
    ).GEO_ID

    states = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age_state.csv"
    ).GEO_ID

    num_districts = len(districts)
    num_states = len(states)

    district_state_codes = [dist_id[9:11] for dist_id in districts]
    state_codes = [state_id[9:11] for state_id in states]

    # Create mapping from state code to state index (position in the states Series)
    state_code_to_idx = {code: idx for idx, code in enumerate(state_codes)}

    # Create indices and values for sparse tensor
    indices = []
    for dist_idx, state_code in enumerate(district_state_codes):
        if state_code in state_code_to_idx:  # Safety check
            state_idx = state_code_to_idx[state_code]
            indices.append([state_idx, dist_idx])

    # Check if we have any valid mappings
    if not indices:
        raise ValueError(
            "No valid district-to-state mappings found. Check the ID formats."
        )

    # Convert to tensors
    indices = torch.tensor(indices, dtype=torch.long).t()
    values = torch.ones(len(indices[0]), dtype=torch.float)

    # Create sparse tensor
    mapping_matrix = torch.sparse.FloatTensor(
        indices, values, torch.Size([num_states, num_districts])
    )

    return mapping_matrix


def create_households(
    sample_per_district: int,
    age_data_by_district: pd.DataFrame,
    target_names: list,
    dataset: str = "cps_2023",
    time_period: int = 2023,
    states = [], # fips
):
    """
    Create household assignments and simulation data needed for metric matrix creation.

    Returns:
        tuple: (households_df, sim_calculations_dict, sim_object)
    """
    sim = Microsimulation(dataset=get_dataset(dataset, time_period))
    sim.default_calculation_period = time_period

    # Extract needed variables from target names
    needed_variables = set()
    for target_name in target_names:
        if target_name.startswith("soi/"):
            # Extract variable name from soi target format
            parts = target_name.split("/")
            if len(parts) >= 3:
                var_name = (
                    parts[2].replace("/count", "").replace("/amount", "")
                )
                needed_variables.add(var_name)

    # Always include AGI for SOI filtering
    needed_variables.add("adjusted_gross_income")

    # Calculate all needed variables
    sim_calculations = {}
    for variable in needed_variables:
        if variable in system.variables:
            values = sim.calculate(variable).values
            values_entity = system.variables[variable].entity.key
            if values_entity == "tax_unit":
                sim_calculations[variable] = values
            else:
                sim_calculations[variable] = sim.map_result(
                    values, values_entity, "tax_unit"
                )

    # Create basic household data DataFrame
    data_by_household = pd.DataFrame(
        {
            "state_fips": sim.calculate("state_fips").values,
            "state_code": sim.calculate("state_code").values,
            "cps_weight": sim.calculate("household_weight").values,
        }
    )

    synth_households = []
    for geo_id in age_data_by_district["GEO_ID"]:
        state_fips_code = int(geo_id[9:11])
        district_code = int(geo_id[11:13])

        if states is not None and state_fips_code not in [int(s) for s in states]:
            continue

        pool = data_by_household[
            data_by_household["state_fips"] == state_fips_code
        ]
        sample_ids = pool.sample(sample_per_district, replace=True).index
        synth_households.append(
            pd.DataFrame(
                {
                    "household_id": sample_ids,
                    "state": state_fips_code,
                    "district": district_code,
                    "weight": data_by_household.loc[
                        sample_ids, "cps_weight"
                    ].values,
                }
            )
        )

    synth_households = pd.concat(synth_households, ignore_index=True)
    return synth_households, sim_calculations, sim


def estimate_targets(weights: torch.Tensor) -> torch.Tensor:
    household_indices = households_tensor[:, 0]
    sampled_household_data = data_by_household_tensor[household_indices]
    weighted_household_data = weights.unsqueeze(1) * sampled_household_data
    estimated_values = weighted_household_data.sum(dim=0)
    return estimated_values

def create_target_normalization_factor() -> torch.Tensor:
    target_names_array = np.array(target_names)

    is_national = np.array(
        ["/national" in name for name in target_names_array]
    )
    is_state = np.array(["/state_" in name for name in target_names_array])
    is_district = np.array(
        ["/district_" in name for name in target_names_array]
    )

    national_factor = is_national * (1 / max(is_national.sum(), 1))
    state_factor = is_state * (1 / max(is_state.sum(), 1))

    district_factor = is_district * (1 / max(is_district.sum(), 1))

    normalization_factor = np.where(
        is_national,
        national_factor,
        np.where(is_state, state_factor, district_factor),
    )

    return torch.tensor(
        normalization_factor, dtype=torch.float32, device=device
    )

   
ParsedT = Tuple[str, str, str, str]  # (source, geography, var_name, var_type)

def build_index_sets(parsed: List[ParsedT],
                     keep_other: bool = False
) -> Dict[str, Dict[Tuple[str, str, str], List[int]]]:
    """
    Group the *indexes* in `parsed` by geography level and by variable.

    Returns a nested dict:
        {
          'national': { (source, var, vtype): [i1, i2, …] , … },
          'state'   : { (source, var, vtype): [ … ], … },
          'district': { … }
        }
    The range portion is *ignored* (because it lives only in target_names).
    """
    level_map = defaultdict(lambda: defaultdict(list))

    for idx, (src, geo, var, vtype) in enumerate(parsed):
        # --- classify the geography -----------------------------
        if geo == "national":
            level = "national"
        elif geo.startswith("state_"):

            level = "state"
        elif geo.startswith("district_"):
            level = "district"
        else:
            if not keep_other:
                continue          # skip Puerto Rico, “other areas”, etc.
            level = "other"       # optional bucket

        # --- accumulate -----------------------------------------
        level_map[level][(src, var, vtype)].append(idx)

    # (optional) make the inner index lists reproducibly ordered
    for lvl in level_map:
        for key in level_map[lvl]:
            level_map[lvl][key].sort()

    return level_map


def get_linear_loss(metrics_matrix, target_vector, sparse=False):
    """Gets the mean squared error loss of X.T @ w wrt y for least squares solution"""
    X = metrics_matrix
    y = target_vector
    if not sparse:
        X_inv_mp = np.linalg.pinv(X)  # Moore-Penrose inverse
        w_mp = X_inv_mp.T @ y
        y_hat = X.T @ w_mp

    else:
        from scipy.sparse import csr_matrix
        from scipy.sparse.linalg import lsqr
        X_sparse = csr_matrix(X)
        result = lsqr(X_sparse.T, y)  # iterative method for sparse matrices
        w_sparse = result[0]
        y_hat = X_sparse.T @ w_sparse

    return round(np.mean((y - y_hat) ** 2), 3)  # mostly for display


# --------------------------------------------------------------------
#                helpers to parse a standardised name
# --------------------------------------------------------------------
def _parse_name(name: str) -> Tuple[str, str, str, str]:
    """
    <source>/<geo id>/<variable name>/<type>/<range>
      └────┬──┘ └───┬───┘ └────┬────┘ └─┬─┘
       src      geo        var_name    typ
    """
    src, geo, var, typ, *_ = name.split('/', 4)
    return src, geo, var, typ


def _geo_level(geo_id: str) -> str:
    if geo_id == 'national':
        return 'national'
    if geo_id.startswith('state_'):
        return 'state'
    if geo_id.startswith('district_'):
        return 'district'
    return 'other'                          # fallback


# --------------------------------------------------------------------
#                          forward transform
# --------------------------------------------------------------------
def scale_targets(target_names: Sequence[str],
                  targets: Sequence[float],
                  *,
                  total_level: float = 100.0,
                  n_states: int = 51,
                  n_districts: int = 436
                 ) -> Tuple[np.ndarray,
                            Dict[Tuple[str, str, str], float]]:
    """
    Standardise every geography so that the *sum* of its rows for a
    given (source, variable, type) is ≈ `total_level`.
    """
    targets = np.asarray(targets, dtype=float)

    parsed = np.array([_parse_name(n) for n in target_names], dtype=object)
    srcs, geos, vars_, typs = parsed.T       # columns

    # -- 1.  base_scale per (source, variable, type) ------------------------
    factors: Dict[Tuple[str, str, str], float] = {}
    for key in {(s, v, t) for s, v, t in zip(srcs, vars_, typs)}:
        s, v, t = key
        mask_nat = (srcs == s) & (vars_ == v) & (typs == t) & (geos == 'national')
        total_nat = targets[mask_nat].sum()
        if total_nat == 0:
            raise ValueError(f"National total is zero for key {key!r}.")
        factors[key] = total_nat / total_level          # denominator

    # -- 2.  apply row‑specific denominators -------------------------------
    scaled = np.empty_like(targets)
    for i, (s, g, v, t, val) in enumerate(zip(srcs, geos, vars_, typs, targets)):
        base = factors[(s, v, t)]
        lvl  = _geo_level(g)
        if   lvl == 'national': denom = base
        elif lvl == 'state':    denom = base / n_states
        elif lvl == 'district': denom = base / n_districts
        else:                   denom = base            # unchanged
        scaled[i] = val / denom

    return scaled, factors

# --------------------------------------------------------------------
#                        inverse transform
# --------------------------------------------------------------------
def unscale_targets(target_names: Sequence[str],
                    scaled_targets: Sequence[float],
                    scale_factors: Dict[Tuple[str, str, str], float],
                    *,
                    n_states: int = 51,
                    n_districts: int = 436
                   ) -> np.ndarray:
    """
    Reconstruct the original targets from `scaled_targets`.
    """
    scaled_targets = np.asarray(scaled_targets, dtype=float)

    parsed = np.array([_parse_name(n) for n in target_names], dtype=object)
    srcs, geos, vars_, typs = parsed.T

    original = np.empty_like(scaled_targets)
    for i, (s, g, v, t, sval) in enumerate(zip(srcs, geos, vars_, typs, scaled_targets)):
        base = scale_factors[(s, v, t)]
        lvl  = _geo_level(g)
        if   lvl == 'national': denom = base
        elif lvl == 'state':    denom = base / n_states
        elif lvl == 'district': denom = base / n_districts
        else:                   denom = base
        original[i] = sval * denom

    return original

# --------------------------------------------------------------------
#                        minimal consistency test
# --------------------------------------------------------------------
def _self_test() -> None:
    rng = np.random.default_rng(0)

    # toy data covering the three geographies for two types
    names = [
        "census/national/age/count/0-4",
        "census/national/age/count/5-9",
        "census/state_tx/age/count/0-4",
        "census/state_tx/age/count/5-9",
        "census/district_tx07/age/count/0-4",
        "census/district_tx07/age/count/5-9",

        "soi/national/adjusted_gross_income/amount/500000_inf",
        "soi/state_tx/adjusted_gross_income/amount/500000_inf",
        "soi/district_tx07/adjusted_gross_income/amount/500000_inf",
    ]

    # fabricate raw numbers: ages (millions / thousands); AGI (billions)
    raw = np.array([
        18_000_000, 20_000_000,                 # nat age
        350_000, 320_000,                       # state age
        42_000, 45_000,                         # district age
        1.2e11, 2.0e9, 4.5e8                    # AGI amounts
    ], dtype=float)

    scaled, factors = scale_targets(names, raw)
    recovered       = unscale_targets(names, scaled, factors)

    assert np.allclose(raw, recovered, rtol=1e-12, atol=1e-9), \
        "round‑trip scaling failed"

    print("✔ self‑test passed – max |Δ| =", np.max(np.abs(raw - recovered)))


# After running functions, start here!! -------------------------------------------------

def calibrate():
    pass


def main():

    # Focus on specific states and districts to reduce data size
    states_list = ["06", "37"]
    states_subsample = True  # or not

    if states_subsample:
        states = states_list
    else:
        states = list(state_abbr_from_fips().keys())
 
    age_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "demographics" / "age.csv"
    )
    agi_data_all_levels = pd.read_csv(
        get_data_directory() / "input" / "soi" / "soi_targets.csv"
    )
   
    # Create regex pattern from states variable
    state_fips_pattern = "|".join(
        [f"0400000US{fips.zfill(2)}" for fips in states]
    )
    district_fips_pattern = "|".join(
        [f"5001800US{fips.zfill(2)}" for fips in states]
    )
    
    combined_pattern = f"^(0100000US|{state_fips_pattern}|{district_fips_pattern})"
    
    age_data_subset = age_data_all_levels[
        age_data_all_levels["GEO_ID"].str.match(combined_pattern)
    ].reset_index(drop=True)
    
    agi_data_subset = agi_data_all_levels[
        agi_data_all_levels["GEO_ID"].str.match(combined_pattern)
    ].reset_index(drop=True)
    
    # Keep district-level data for household creation logic
    age_data_by_district = age_data_subset.loc[
        lambda df: df["GEO_ID"].str.startswith("5001800US")
    ].reset_index(drop=True)
    
    # Create target matrix
    targets = create_target_matrix(age_data_subset, agi_data_subset)
    target_names = list(targets.columns)
    
    # Create households and simulation data based on target requirements
    households, sim_calculations, sim = create_households(
        sample_per_district=500,
        age_data_by_district=age_data_by_district,
        target_names=target_names,
        dataset="cps_2023",
        time_period=2023,
        states=states,
    )
    
    # Create metric matrix
    data_by_household = create_metric_matrix(
        sim=sim,
        sim_calculations=sim_calculations,
        ages=age_data_subset,
        soi_targets=agi_data_subset,
        households=households,
    )

    weights = households["weight"].to_numpy(copy=True)
    
    device = "mps:0" if torch.backends.mps.is_available() else "cpu"
    
    data_by_household_tensor = torch.tensor(
        data_by_household.drop(columns=["state_code", "state_fips"])
        .astype(float)
        .values,
        dtype=torch.float32,
        device=device,
    )
    households_tensor = torch.tensor(
        households.values, dtype=torch.int64, device=device
    )
    targets = targets.values.flatten()
    
    parsed = [_parse_name(n) for n in target_names]
    
    ## Do the analysis ----
    use_sparse = True
    household_indices = households_tensor[:, 0]
    sampled_household_data = data_by_household_tensor[household_indices]
    
    scaled_targets, factors = scale_targets(target_names, targets)
    
    X = sampled_household_data.numpy()
    y = scaled_targets
    
    idx_dict = build_index_sets(parsed)
    ordered_slices  = []
    cumulative_slices = []
    for level in ("national", "state", "district"):
        for var_key, index_list in idx_dict[level].items():
            ordered_slices.append(index_list)
            cumulative_slices.extend(index_list)
            loss = get_linear_loss(X[:, cumulative_slices], scaled_targets[cumulative_slices], use_sparse)
            print(f"Level: {level}. Adding: {'/'.join(var_key)}, Loss: {loss}")
    
    
    # Calibration ------- 
    calibration = Calibration(
        weights=40 * torch.ones(weights.shape[0]),
        targets=scaled_targets,
        target_names=target_names,
        estimate_function=estimate_targets,
        epochs=20,
        noise_level=0,
        learning_rate=.001,
        dropout_rate = 0.0,
        normalization_factor = torch.ones(1908),
        device = "cpu"
    )
    
    calibration.calibrate()
    calibration.performance_df.to_csv("calibration_log.csv", index=False)
    
    
