from datetime import date

from policyengine_us import Microsimulation

sim = Microsimulation(
    dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",
)
sim.macro_cache_read = False
sim.macro_cache_write = False
sim.opt_out_cache = True



%%time
sim.calculate("income_tax").sum() / 1E9  # 1663


%%time
sim.calculate("income_tax").sum() / 1E9  # 1663


from pathlib import Path
import shutil

# Delete the actual cache folder
data_dir = Path(sim.data_storage_dir)
for cache_folder in data_dir.glob("*_variable_cache"):
    shutil.rmtree(cache_folder)

# Clear in-memory caches
for computed_variable in sim.tax_benefit_system.variables:
    if computed_variable not in sim.input_variables:
        sim.delete_arrays(computed_variable)

# Check what's in the on-disk storage
for holder in sim.populations['person']._holders.values():
    if holder._disk_storage:
        print(f"{holder.variable.name}: has disk storage at {holder._disk_storage.storage_dir}")
        holder._disk_storage.delete()  # Clear it


  # First time
sim1 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")

%%time
sim1.daaset.load(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5",

sim1.calculate("income_tax").sum() / 1E9

sim2 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")

%%time
sim2.calculate("income_tax").sum() / 1E9





%%time
sim.calculate("income_tax").sum() / 1E9  # 1663

%%time
sim.calculate("income_tax").sum() / 1E9  # 1663



for computed_variable in sim.tax_benefit_system.variables:
    if computed_variable not in sim.input_variables:
        sim.delete_arrays(computed_variable)



hh_weights = sim.calculate("income_tax", map_to="household").weights.values
person_weights = sim.calculate("income_tax", map_to="person").weights.values

sim.calculate("income_tax").sum() / 1E9  # 1663
sim.calculate("income_tax", 2023).sum() / 1E9  # 1663

sim2 = Microsimulation(dataset="hf://policyengine/policyengine-us-data/cps_2023.h5")
sim2.set_input("household_weight", 2023, hh_weights * 100)
sim2.set_input("person_weight", 2023, person_weights * 100)

sim2.default_calculation_period
sim2.calculate("income_tax").sum() / 1E9  # 1663336
sim2.calculate("income_tax", 2023).sum() / 1E9  # 1663336

for computed_variable in sim2.tax_benefit_system.variables:
    if computed_variable not in sim2.input_variables:
        sim2.delete_arrays(computed_variable)


parameters = sim.tax_benefit_system.parameters

# Get uprating index
uprating_2023 = parameters("2023-01-01").gov.ssa.uprating
uprating_2035 = parameters("2035-01-01").gov.ssa.uprating
uprating_2100 = parameters("2100-01-01").gov.ssa.uprating

print(f"SSA Uprating Index:")
print(f"2023: {uprating_2023}")
print(f"2035: {uprating_2035}")
print(f"2100: {uprating_2100}")

# Get raw CPI-W (Q3 average)

cpi_w_2023_q3 = sum([
    parameters(date(2023, m, 1)).gov.bls.cpi.cpi_w
    for m in [7, 8, 9]
]) / 3
cpi_w_2035_q3 = sum([
    parameters(date(2035, m, 1)).gov.bls.cpi.cpi_w
    for m in [7, 8, 9]
]) / 3
cpi_w_2100_q3 = sum([
    parameters(date(2100, m, 1)).gov.bls.cpi.cpi_w
    for m in [7, 8, 9]
]) / 3

print(f"\nRaw CPI-W (Q3 average):")
print(f"2023: {cpi_w_2023_q3:.2f}")
print(f"2035: {cpi_w_2035_q3:.2f}")
print(f"2100: {cpi_w_2100_q3:.2f}")

print(f"\nCPI-W ratio 2100/2023: {cpi_w_2100_q3 / cpi_w_2023_q3:.3f}")
print(f"Uprating ratio 2100/2023: {uprating_2100 / uprating_2023:.3f}")

params = sim.tax_benefit_system.parameters

pop_2023 = params('2023-01-01').calibration.gov.census.populations.total
pop_2100 = params('2100-01-01').calibration.gov.census.populations.total

pop_growth_factor = pop_2100 / pop_2023
print(f"Population growth 2023-2100: {pop_growth_factor:.4f}x")


# Social Security ----
#class social_security(Variable):
#    value_type = float
#    entity = Person
#    definition_period = YEAR
#    documentation = "Social Security benefits, not including SSI"
#    label = "Social Security"
#    unit = USD
#    adds = [
#        "social_security_" + i
#        for i in ["dependents", "disability", "retirement", "survivors"]
#    ]
#    uprating = "gov.ssa.uprating"

# So it's a person-level entity variable that uses the gov.ssa.uprating, which
# should be CPI-W from the previous year's 3rd quarter
sim.parameters.gov.ssa.uprating('2023-01-01')

sim.calculate("social_security", period = 2023, map_to="person").sum() / 1E9
sim.calculate("social_security", period = 2025, map_to="person").sum() / 1E9
sim.calculate("social_security", period = 2035, map_to="person").sum() / 1E9
sim.calculate("social_security", period = 2054, map_to="person").sum() / 1E9
sim.calculate("social_security", period = 2100, map_to="person").sum() / 1E9


# Changing weights dynamically. I'm off topic now .....
sim.set_input("person_weight", 2023, np.repeat(1, 50863))
sim.calculate("social_security", period = 2023, map_to="person").sum() / 1E9

holder = sim.get_holder("person_weight")
holder.get_array('2023')
holder.delete_arrays('2023')
holder.get_array('2023')

holder.set_input(2023,np.repeat(1, 50863))


# 1127 is a little low for 2023

1609 / 1200.6  # for 2025, we're off by 1.34x, so that's just poor calibration, so we can adjust

# Let's manually uprate it to 2100
1.34 * 1127 * uprating_2100 / uprating_2023 * pop_growth_factor

# At 2035, we're up to 1614
# The CPI-W projection for 2034, which 2035's SS benefit would have been based on, is 123.91
# The CPI-W projection for 2099, which 2100's SS benefit would be based on, is 578.89
# The manual CPI-W projection from 2035 to 2100 is:
(578.89 / 123.91) * 1614  # 7540.4

# That's less than the 9412 that sim.calculate projects

# The Trustees' intermediate scenario: Social Security benefits will cost 5,809 Billion in 2100 in 2025 dollars
# Adjusted for inflation, that's
5809 * (592.78 / 100)
# Or $34,345 billion, so we're off by a factor of over 3x even with our higher sim.calcuate than CPI-W would project



# This strongly suggests double-uprating: the components are being uprated once by
# uprating = "gov.ssa.uprating", and then again by something else.

# And the NaN for Dependents/Survivors is odd - those should have values. Are they
# actually zero in the microdata, or is something else going on?

sim.calculate("social_security_survivors", period=2023, map_to="person").sum() 
sim.calculate("social_security_dependents", period=2023, map_to="person").sum() 

# It's interesting that the uprating is commented out

# spm_unit_weight

['marital_unit_weight', 'spm_unit_weight', 'family_weight', 'household_weight', 'tax_unit_weight', 'person_weight']
