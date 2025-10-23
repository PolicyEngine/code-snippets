import pandas as pd

from policyengine_us import Microsimulation

sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

period = 2024

df_flat = pd.DataFrame({
    'household_id': sim.calculate('household_id', period=period, map_to="person"),
    'household_weight': sim.calculate('household_weight', period=period, map_to="person"),
    'person_id': sim.calculate('person_id', period=period, map_to="person"),
    'tax_unit_id': sim.calculate('tax_unit_id', period=period, map_to="person"),
})

# NOTE:
# Be careful, as variables at the househould level and tax_unit level will be repeated
# That includes household_weight
# You are in charge of how to aggregate to the household level to use the weights.
