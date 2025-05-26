import numpy as np
from numpy import where

from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS_2024

from policyengine_core.periods import period as period_  # alias, "period" will be a var
from policyengine_core.populations.population import Population
from policyengine_core.entities import Entity
from policyengine_core.reforms import Reform

from policyengine_us import Microsimulation
from policyengine_us.model_api import add, max_, min_


# Create a simulation as we will harvest its internals
sim = Microsimulation(dataset=EnhancedCPS_2024)
year = 2022
period = period_(year)

# Entity init arguments are: the key, the plural, the label, the doc string
Person = Entity("person", "people", "Person", "A person")
print(Person.key)

# Population: Lowercase p person
person = Population(Person)
print(f"The person population is a population of {person.entity.plural}")
person.simulation = sim  # populations need simulations
person.entity.set_tax_benefit_system(sim.tax_benefit_system)  # and a tax benefit system through their entity

parameters = sim.tax_benefit_system.parameters

p = parameters(period).gov.irs.deductions.qbi

# 1. Table 2, Panel A, Entity 1, as is

# 1. Table 2, Panel A, Entity 1, income set to 20K over upper threshold 
# 2. Table 2, Panel A, Entity 1, income set to 10K under lower threshold 
# 3. Table 2, Panel A, Entity 4, as is
# 4. Table 2, Panel A, Entity 4, income set to 20K over upper threshold 
# 5. Table 2, Panel A, Entity 4, income set to 10K under lower threshold 
# 6. O3 example where qbi is large but taxible income is modest 

# The original qbid_amount.py
#qbi = person("qualified_business_income", period)
qbi = np.array([80_000, 80_000, 80_000, 110_000, 110_000, 110_000, 200_000])

qbid_max = p.max.rate * qbi  # Worksheet 12-A, line 3
# compute caps
# w2_wages = person("w2_wages_from_qualified_business", period)
w2_wages = np.array([8_000, 8_000, 8_000, 60_000, 60_000, 60_000, 150_000])
# b_property = person("unadjusted_basis_qualified_property", period)
b_property = np.array([50_000, 50_000, 50_000, 6_000, 6_000, 6_000, 0])
wage_cap = w2_wages * p.max.w2_wages.rate  # Worksheet 12-A, line 5
alt_cap = (  # Worksheet 12-A, line 9
    w2_wages * p.max.w2_wages.alt_rate
    + b_property * p.max.business_property.rate
)
full_cap = max_(wage_cap, alt_cap)  # Worksheet 12-A, line 10
# compute phase-out ranges
# taxinc_less_qbid = person.tax_unit("taxable_income_less_qbid", period)
taxinc_less_qbid = np.array([400_100, 460_100, 330_100, 400_100, 460_100, 330_100, 120_000])

# filing_status = person.tax_unit("filing_status", period)
# po_start = p.phase_out.start[filing_status]
# po_length = p.phase_out.length[filing_status]

# Joint filer 2022
po_start = np.repeat(340_100, qbi.shape[0])
po_length = np.repeat(100_000, qbi.shape[0])

# compute phase-out limited QBID amount
reduction_rate = min_(  # Worksheet 12-A, line 24; Schedule A, line 9
    1, (max_(0, taxinc_less_qbid - po_start)) / po_length
)
applicable_rate = 1 - reduction_rate  # Schedule A, line 10

#is_sstb = person("business_is_sstb", period)
is_sstb = np.array([0, 0, 0, 1, 1, 1, 0])
# Schedule A, line 11
sstb_multiplier = where(is_sstb, applicable_rate, 1)
adj_qbid_max = qbid_max * sstb_multiplier
# Schedule A, line 12 and line 13
adj_cap = full_cap * sstb_multiplier
line11 = min_(adj_qbid_max, adj_cap)  # Worksheet 12-A, line 11
# NOTE: the above logic is just applying a multiplier if it's an SSTB, not sure why

# compute phased reduction
reduction = reduction_rate * max_(  # Worksheet 12-A, line 25
    0, adj_qbid_max - adj_cap
)
line26 = max_(0, adj_qbid_max - reduction)

# Asking whether the wage and property amount is less than qbi * 20%, if so, using adj qbi * 20%. Why?
line12 = where(adj_cap < adj_qbid_max, line26, 0)   # Awkward mechanism to chose reduced qbi * 20% the next line if prop cap is larger 
max_(line11, line12)  # Worksheet 12-A, line 13


# Error in the original code on 7:

# The dataframe shown in the canvas highlights the $16 000 overstatement.
# 
# Why it happens:
# 
# Below the W-2/UBIA threshold: both versions start with 20 % × QBI → $40 000.
# 
# Overall 20 % income cap:
# Statute: min(deduction so far, 20 % × taxable income) → min($40 000, $24 000) = $24 000.
# Original code: skips this check → allows the full $40 000, violating §199A(a)(2).

# Get's the right answer on a Non-SSTB within the phase in range, but:
# SSTB multiplier seems too simple
# Unclear how this works outside of the threshold

# 3 What else you might need
# Overall 20 % taxable-income cap
# After you aggregate all businesses and REIT/PTP amounts, §199A(a)(2) limits the deduction to 20 % of (taxable income – net capital gain). That happens later on Form 8995-A; if your engine applies it elsewhere, you’re good—just be sure it’s in the pipeline.
# 
# Carry-forwards & loss netting
# Schedule C net-loss carry-forwards, suspended losses, patron reductions, etc., are outside this block. If you track them elsewhere, fine; otherwise, add them.
# 
# Aggregation & multi-business taxpayers
# The snippet is business-level. When you aggregate businesses you’ll need to apply the wage/UBIA limit and phase-in at the aggregation level, then allocate the allowed deduction back to owners.


# Now, let's do qualified business income within:
# variables/gov/irs/income/taxable_income/.../qualified_business_income.py

p = parameters(period).gov.irs.deductions.qbi
gross_qbi = add(person, period, p.income_definition)
qbi_deductions = add(person, period, p.deduction_definition)
qbi = max_(0, gross_qbi - qbi_deductions)  # the return

len(qbi)

w = sim.calculate("person_weight", map_to="person", period=year)
assert(len(w) == len(qbi))

print(np.dot(qbi, w)  / 1E9)
# Compare to:
sim.calculate("qualified_business_income", map_to="household", period=year).sum() / 1E9


# taxinc_less_qbid

# Entity init arguments are: the key, the plural, the label, the doc string
TaxUnit = Entity("tax_unit", "tax_units", "Tax Unit", "A tax unit")
print(TaxUnit.key)

# Population: Lowercase p person
tax_unit = Population(TaxUnit)
tax_unit.simulation = sim  # populations need simulations
tax_unit.entity.set_tax_benefit_system(sim.tax_benefit_system)  # and a tax benefit system through their entity
year = 2029
period = period_(year)

parameters = sim.tax_benefit_system.parameters

p = parameters(period).gov.irs.deductions.qbi

agi = tax_unit("adjusted_gross_income", period)
p = parameters(period).gov.irs.deductions
ded_if_itemizing = [
    deduction
    for deduction in p.deductions_if_itemizing
    if deduction != "qualified_business_income_deduction"
]
ded_if_not_itemizing = [
    deduction
    for deduction in p.deductions_if_not_itemizing
    if deduction != "qualified_business_income_deduction"
]
ded_value_if_itemizing = add(tax_unit, period, ded_if_itemizing)
ded_value_if_not_itemizing = add(
    tax_unit, period, ded_if_not_itemizing
)
itemizes = tax_unit("tax_unit_itemizes", period)
ded_value = where(
    itemizes,
    ded_value_if_itemizing,
    ded_value_if_not_itemizing,
)
max_(0, agi - ded_value)   # return
