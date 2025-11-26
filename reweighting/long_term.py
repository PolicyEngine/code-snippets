import os

import pandas as pd
import numpy as np

from policyengine_us import Microsimulation

#H5_PATH = 'hf://policyengine/test/'
H5_PATH = '/home/baogorek/devl/sep/policyengine-us-data/policyengine_us_data/datasets/cps/long_term/projected_datasets/'

# 2027 --------------------------------------
sim = Microsimulation(dataset = H5_PATH + "2027.h5")
parameters = sim.tax_benefit_system.parameters

## Total social security
assert sim.default_calculation_period == '2027'
ss_estimate_cost_b = sim.calculate("social_security").sum() / 1E9

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G10 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2027, Cost is: $1,715 billion
ss_trustees_cost_b = 1_800
assert round(ss_estimate_cost_b) == ss_trustees_cost_b

## Taxable Payroll for Social Security
taxible_estimate_b = (
    sim.calculate("taxable_earnings_for_social_security").sum() / 1E9
    + sim.calculate("social_security_taxable_self_employment_income").sum() / 1E9
)

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2027, Cost is: $1,715 billion
ss_trustees_payroll_b = 11_627
assert round(taxible_estimate_b) == ss_trustees_payroll_b

## Population demographics, total

### Population count of 6 year olds
person_weights = sim.calculate("age", map_to="person").weights
person_ages = sim.calculate("age", map_to="person").values
person_is_6 = person_ages == 6

total_age6_est = np.sum(person_is_6 * person_weights)

### Single Year Age demographic projections - latest published is 2024:
### "Mid Year" CSV from https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html
### Row 8694, Col C, contains 3730632
ss_age6_pop = 3_730_632
assert ss_age6_pop == round(total_age6_est)


# 2100 --------------------------------------
sim = Microsimulation(dataset = H5_PATH + "2100.h5")
parameters = sim.tax_benefit_system.parameters

## Total social security
assert sim.default_calculation_period == '2100'
ss_estimate_cost_b = sim.calculate("social_security").sum() / 1E9

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G10 (nominal dollars in billions)
### Intermediate scenario for row 69, for Intermediate Scenario, 2100, Cost is: $34,432 billion
ss_trustees_cost_b = 34_432 
# Rounding takes it off a bit. We calibrated to CPI ratio times 2025-dollar cost
assert np.allclose(ss_estimate_cost_b, ss_trustees_cost_b, rtol=.0001)

## Taxable Payroll for Social Security
taxible_estimate_b = (
    sim.calculate("taxable_earnings_for_social_security").sum() / 1E9
    + sim.calculate("social_security_taxable_self_employment_income").sum() / 1E9
)

### Trustees SingleYearTRTables_TR2025.xlsx, Tab VI.G6 (nominal dollars in billions)
### Intermediate scenario for row 143, for Intermediate Scenario, 2100, Cost is: $187,614 billion
ss_trustees_payroll_b = 187_614
assert round(taxible_estimate_b) == ss_trustees_payroll_b

## Population demographics, total

### Population count of 6 year olds
person_weights = sim.calculate("age", map_to="person").weights
person_ages = sim.calculate("age", map_to="person").values
person_is_6 = person_ages == 6

total_age6_est = np.sum(person_is_6 * person_weights)

### Single Year Age demographic projections - latest published is 2024:
### "Mid Year" CSV from https://www.ssa.gov/oact/HistEst/Population/2024/Population2024.html
### Row 16067, Col C, contains 5162540
ss_age6_pop = 5_162_540
assert ss_age6_pop == round(total_age6_est)
