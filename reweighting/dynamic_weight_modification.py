"""
Dynamic Weight Modification in PolicyEngine Microsimulation

When changing household_weight, ALL derived weight variables must be invalidated
"""

import numpy as np
from policyengine_us import Microsimulation

sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
period = 2024
n_households = len(sim.calculate("household_id", period))

# Calculate total income tax with original weights
income_tax = sim.calculate("income_tax", period)
sim.calculate("income_tax", period)
income_tax.sum() / 1E9


# Change weights and nothing happens!
new_weights = 10 * np.ones(n_households)
sim.set_input("household_weight", period, new_weights)

sim.calculate("income_tax", period)  # weights not changed
income_tax = sim.calculate("income_tax", period)
income_tax.sum() / 1E9  # sum not changed

# Invalidate all weight-related caches so they recalculate
for entity_weight in ['household_weight', 'tax_unit_weight', 'person_weight',
                      'spm_unit_weight', 'marital_unit_weight']:
    sim.delete_arrays(entity_weight, period)

# Recalculate with new weights
sim.calculate("income_tax", period)  # Now they're zeros
sim.calculate("income_tax", period, map_to="household")  # Now they're zeros
sim.calculate("income_tax", period).sum() / 1E9  # zero, because tax_unit weights are zero
sim.calculate("income_tax", period, map_to="household").sum() / 1E9  # not zero, because household weights are zero

income_tax = sim.calculate("income_tax", period).sum()
income_tax.sum() / 1E9
