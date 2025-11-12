import pandas as pd
import numpy as np

from policyengine_us import Microsimulation

# Let's build some h5s and really figure out how SNAP aggregation works


# The original Census Bureau variable is SPMSNAP from the Annual Social and Economic Supplement (ASEC)

# https://cps.ipums.org/cps-action/variables/SPMSNAP

sim = Microsimulation(
    dataset="/home/baogorek/devl/policyengine-us-data/policyengine_us_data/storage/cps_2024_full.h5"
)

sim.calculate("snap_reported").sum() / 1E9  # 44.4
sim.calculate("snap_reported", map_to="person").sum() / 1E9  # 143.5, or about 3x, from the broadcast
sim.calculate("snap_reported", map_to="spm_unit").sum() / 1E9  # 44.4 
sim.calculate("snap_reported", map_to="household").sum() / 1E9  # 44.4 
sim.calculate("snap_reported", map_to="tax_unit").sum() / 1E9  # 44.4 
sim.calculate("snap_reported", map_to="family").sum() / 1E9  # 0
sim.calculate("snap_reported", map_to="marital_unit").sum() / 1E9  # 44.4


# Note that the snap variable has takeup rate logic built in, and that's what the
# "takeup seed is all about"

# takes_up_snap_if_eligible.py (lines 10-13):
# 
# def formula(spm_unit, period, parameters):
#     seed = spm_unit("snap_take_up_seed", period)
#     takeup_rate = parameters(period).gov.usda.snap.takeup_rate
#     return seed < takeup_rate
# 
# So the mechanics:
# 
# 1. snap_take_up_seed = random number [0, 1) for each SPM unit (created in your CPS data at cps.py:226)
# 2. takeup_rate = a parameter (likely ~70-80% based on empirical SNAP takeup rates)
# 3. Returns: True if seed < takeup_rate, False otherwise



sim.calculate("snap").sum() / 1E9  # 50.3
sim.calculate("snap", map_to="person").sum() / 1E9  # 155.6, or about 3x, from the broadcast
sim.calculate("snap", map_to="spm_unit").sum() / 1E9  # 50.3 
sim.calculate("snap", map_to="household").sum() / 1E9  # 50.3 
sim.calculate("snap", map_to="tax_unit").sum() / 1E9  # 50.3 
sim.calculate("snap", map_to="family").sum() / 1E9  # 0
sim.calculate("snap", map_to="marital_unit").sum() / 1E9  # 50.3

snap_df = sim.calculate_dataframe(["spm_unit_id", "snap"], map_to="spm_unit")
# 58147 spm_units
snap_df.loc[snap_df.snap > 10000]
404001

entity_rel = pd.DataFrame({
    'person_id': sim.calculate("person_id"),
    'person_household_id': sim.calculate("person_household_id"),
    'person_spm_unit_id': sim.calculate("person_spm_unit_id")
})

merged_df = entity_rel.merge(
    snap_df, 
    left_on="person_spm_unit_id", 
    right_on="spm_unit_id"
)

merged_df.loc[merged_df.person_spm_unit_id == 404001]

# Oh hell, snap is definitely broadcast to the person level
