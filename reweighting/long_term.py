import os

import pandas as pd
import numpy as np

from policyengine_us import Microsimulation

H5_PATH = '/home/baogorek/devl/sep/policyengine-us-data/policyengine_us_data/datasets/cps/long_term/projected_datasets/'

# 2027
sim = Microsimulation(dataset = H5_PATH + "2027.h5")
parameters = sim.tax_benefit_system.parameters

assert sim.default_calculation_period == '2027'
ss_total_b = sim.calculate("social_security").sum() / 1E9

# Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G9
# Intermediate scenario for row 69, for Intermediate Scenario, 2027, Cost is: $1,715 billion
ss_cost_b = 1_715
assert ss_total_b > ss_cost_b  # 2 years of inflation


# Note:  not our CPI-W: parameters.gov.bls.cpi.cpi_w("2026-01-05")
#
# CPI from Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6
cpi_w_2025 = 100
cpi_w_2027 = 104.95

cpi_w_2025_b = parameters.gov.ssa.uprating("2025-01-01")
cpi_w_2027_b = parameters.gov.ssa.uprating("2027-01-01")

ratio = cpi_w_2027 / cpi_w_2025
ratio_b = cpi_w_2027_b / cpi_w_2025_b

assert round(ss_total_b) == round(ss_cost_b * ratio)  # Fails, but close


# 2100
sim = Microsimulation(dataset = H5_PATH + "2100.h5")
parameters = sim.tax_benefit_system.parameters

assert sim.default_calculation_period == '2100'
ss_total_b = sim.calculate("social_security").sum() / 1E9

# Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G9
# Intermediate scenario for row 143, for Intermediate Scenario, 2100, Cost is: $1,033,686.26 billion
ss_cost_b = 5809
assert ss_total_b > ss_cost_b  # many years of inflation

parameters.gov.ssa.uprating# Note:  not our CPI-W: parameters.gov.bls.cpi.cpi_w("2026-01-05")
#
# CPI from Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6
cpi_w_2025 = 100
cpi_w_2100 = 592.78

cpi_w_2025_b = parameters.gov.ssa.uprating("2025-01-06")
cpi_w_2100_b = parameters.gov.ssa.uprating("2100-01-06")

ratio = cpi_w_2100 / cpi_w_2025
ratio_b = cpi_w_2100_b / cpi_w_2025_b

assert round(ss_total_b) == round(ss_cost_b * ratio)  # fails, not close!

# Population count, total
ss_total_pop = 458_325_282
total_pop_est = np.sum(sim.calculate("person_weight", map_to="person").weights)
assert round(ss_total_pop) == round(total_pop_est)

# Population count of 6 year olds
ss_age6_pop = 5_162_540

person_weights = sim.calculate("age", map_to="person").weights
person_ages = sim.calculate("age", map_to="person").values
person_is_6 = person_ages == 6

total_age6_est = np.sum(person_is_6 * person_weights)
assert ss_age6_pop == round(total_age6_est)
