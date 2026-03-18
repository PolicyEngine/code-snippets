import numpy as np
from policyengine_us import Microsimulation

DATASET = "hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5"
YEAR = 2024

sim = Microsimulation(dataset=DATASET)

tu_w = sim.calculate("household_weight", period=YEAR, map_to="tax_unit").values
aca = sim.calculate("aca_ptc", period=YEAR, map_to="tax_unit").values
filer = sim.calculate("tax_unit_is_filer", period=YEAR, map_to="tax_unit").values.astype(bool)

pos = aca > 0
print("All tax units with aca_ptc > 0:")
print("  total aca_ptc", float(np.sum(aca[pos] * tu_w[pos])))
print("  count", float(np.sum(tu_w[pos])))

mask = pos & filer
print("Filers only with aca_ptc > 0:")
print("  total aca_ptc", float(np.sum(aca[mask] * tu_w[mask])))
print("  count", float(np.sum(tu_w[mask])))
