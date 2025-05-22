import numpy as np
from numpy import where

from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS_2024

from policyengine_core.periods import period as period_  # alias, "period" will be a var
from policyengine_core.populations.population import Population
from policyengine_core.entities import Entity
from policyengine_core.reforms import Reform

from policyengine_us import Microsimulation
from policyengine_us.model_api import add, max_, min_


qbid_reform = Reform.from_dict({
  "gov.irs.deductions.qbi.max.rate": {
    "2026-01-01.2100-12-31": 0.23
  },
  "gov.irs.deductions.qbi.max.w2_wages.rate": {
    "2026-01-01.2100-12-31": 0.5
  },
  "gov.irs.deductions.qbi.max.w2_wages.alt_rate": {
    "2026-01-01.2035-12-31": 0.25
  },
  "gov.irs.deductions.qbi.max.business_property.rate": {
    "2026-01-01.2100-12-31": 0.025
  },
  "gov.contrib.reconciliation.qbid.phase_out_rate": {
    "2026-01-01.2100-12-31": 0.75
  },
  "gov.irs.deductions.qbi.phase_out.length.HEAD_OF_HOUSEHOLD": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.JOINT": {
    "2026-01-01.2100-12-31": 100_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SEPARATE": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SINGLE": {
    "2026-01-01.2100-12-31": 50_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.length.SURVIVING_SPOUSE": {
    "2026-01-01.2100-12-31": 100_000.0 
  },
  "gov.irs.deductions.qbi.phase_out.start.JOINT": {
    "2026-01-01.2026-12-31": 400600,
    "2027-01-01.2027-12-31": 410500,
    "2028-01-01.2028-12-31": 419000,
    "2029-01-01.2029-12-31": 427350,
    "2030-01-01.2030-12-31": 435900,
    "2031-01-01.2031-12-31": 444500,
    "2032-01-01.2032-12-31": 453250,
    "2033-01-01.2033-12-31": 462200,
    "2034-01-01.2034-12-31": 471400,
    "2035-01-01.2036-12-31": 480700
  },
  "gov.irs.deductions.qbi.phase_out.start.SINGLE": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217900,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  },
  "gov.irs.deductions.qbi.phase_out.start.SEPARATE": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217950,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  },
  "gov.irs.deductions.qbi.phase_out.start.SURVIVING_SPOUSE": {
    "2026-01-01.2026-12-31": 400600,
    "2027-01-01.2027-12-31": 410500,
    "2028-01-01.2028-12-31": 419000,
    "2029-01-01.2029-12-31": 427350,
    "2030-01-01.2030-12-31": 435900,
    "2031-01-01.2031-12-31": 444500,
    "2032-01-01.2032-12-31": 453250,
    "2033-01-01.2033-12-31": 462200,
    "2034-01-01.2034-12-31": 471400,
    "2035-01-01.2036-12-31": 480700
  },
  "gov.irs.deductions.qbi.phase_out.start.HEAD_OF_HOUSEHOLD": {
    "2026-01-01.2026-12-31": 200300,
    "2027-01-01.2027-12-31": 205250,
    "2028-01-01.2028-12-31": 209500,
    "2029-01-01.2029-12-31": 213650,
    "2030-01-01.2030-12-31": 217950,
    "2031-01-01.2031-12-31": 222250,
    "2032-01-01.2032-12-31": 226600,
    "2033-01-01.2033-12-31": 231100,
    "2034-01-01.2034-12-31": 235700,
    "2035-01-01.2036-12-31": 240350
  }
}, country_id="us")


# Create a simulation as we will harvest its internals
sim = Microsimulation(dataset=EnhancedCPS_2024, reform=qbid_reform)
year = 2029
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


# First, let's do qualified business income within:
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

# My analysis  ---

# ---- and NOW you can start to run the formula in: 
# /policyengine_us/reforms/reconciliation/reconciled_qbid.py  ------- 
reform = True
if reform == True:  
    p = parameters(period).gov.irs.deductions
    
    # 1. Compute the new maximum QBID
    # Grab this from the interactive code above
    #qbi = person("qualified_business_income", period)
    qbid_max = p.qbi.max.rate * qbi
    
    # 2. Compute the wage/property cap (unchanged)
    w2_wages = person("w2_wages_from_qualified_business", period)
    b_property = person("unadjusted_basis_qualified_property", period)
    wage_cap = w2_wages * p.qbi.max.w2_wages.rate
    alt_cap = (
        w2_wages * p.qbi.max.w2_wages.alt_rate
        + b_property * p.qbi.max.business_property.rate
    )
    full_cap = max_(wage_cap, alt_cap)
    
    # 3. Phase-out logic: 75% of each dollar above the threshold
    taxinc_less_qbid = person.tax_unit(
        "taxable_income_less_qbid", period
    )
    filing_status = person.tax_unit("filing_status", period)
    threshold = p.qbi.phase_out.start[filing_status]
    p_ref = parameters(period).gov.contrib.reconciliation.qbid
    phase_out_rate = p_ref.phase_out_rate 
    excess_income = max_(0, taxinc_less_qbid - threshold)
    reduction_amount = phase_out_rate * excess_income
    # 4. Apply phase-out to the deduction
    phased_deduction = max_(0, qbid_max - reduction_amount)
    # 5. Final QBID is the lesser of the phased deduction or the wage/property cap
    qbid = min_(phased_deduction, full_cap)

    # Note that qbid is a vector
    len(qbid)
    assert(len(w) == len(qbid))
    print(np.dot(qbid, w)  / 1E9)

    len(qbid2)
    print(np.dot(qbid2, w)  / 1E9)

    # Compare to:
    sim.calculate("qbid_amount", map_to="household", period=year).sum() / 1E9
 
else:
    p = parameters(period).gov.irs.deductions.qbi
    # Grab this from the interactive code above
    #qbi = person("qualified_business_income", period)
    qbid_max = p.max.rate * qbi  # Worksheet 12-A, line 3
    # compute caps
    w2_wages = person("w2_wages_from_qualified_business", period)
    b_property = person("unadjusted_basis_qualified_property", period)
    wage_cap = w2_wages * p.max.w2_wages.rate  # Worksheet 12-A, line 5
    alt_cap = (  # Worksheet 12-A, line 9
        w2_wages * p.max.w2_wages.alt_rate
        + b_property * p.max.business_property.rate
    )
    full_cap = max_(wage_cap, alt_cap)  # Worksheet 12-A, line 10
    # compute phase-out ranges
    taxinc_less_qbid = person.tax_unit("taxable_income_less_qbid", period)
    filing_status = person.tax_unit("filing_status", period)
    po_start = p.phase_out.start[filing_status]
    po_length = p.phase_out.length[filing_status]
    # compute phase-out limited QBID amount
    reduction_rate = min_(  # Worksheet 12-A, line 24; Schedule A, line 9
        1, (max_(0, taxinc_less_qbid - po_start)) / po_length
    )  # mean of .16
    applicable_rate = 1 - reduction_rate  # Schedule A, line 10
    is_sstb = person("business_is_sstb", period)
    # Schedule A, line 11
    sstb_multiplier = where(is_sstb, applicable_rate, 1)
    adj_qbid_max = qbid_max * sstb_multiplier
    # Schedule A, line 12 and line 13
    adj_cap = full_cap * sstb_multiplier
    line11 = min_(adj_qbid_max, adj_cap)  # Worksheet 12-A, line 11
    # compute phased reduction
    reduction = reduction_rate * max_(  # Worksheet 12-A, line 25
        0, adj_qbid_max - adj_cap
    )
    line26 = max_(0, adj_qbid_max - reduction)
    line12 = where(adj_cap < adj_qbid_max, line26, 0)
    qbid = max_(line11, line12)  # Worksheet 12-A, line 13
   
    print(np.dot(qbid, w)  / 1E9)
