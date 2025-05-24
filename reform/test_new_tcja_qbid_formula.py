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
year = 2021
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



from policyengine_us.model_api import *

# ————————————————— 1.  Basic inputs ————————————————————————————
qbi                  = person("qualified_business_income", period)
w2_wages             = person("w2_wages_from_qualified_business", period)
ubia_property        = person("unadjusted_basis_qualified_property", period)
is_sstb              = person("business_is_sstb", period)
reit_ptp_income      = person("qualified_reit_and_ptp_income", period)

tax_inc_before_qbid  = person.tax_unit("taxable_income_less_qbid", period)
filing_status        = person.tax_unit("filing_status", period)

# Thresholds and ranges keyed by filing status come from parameters
income_threshold     = p.phase_out.start[filing_status]
phase_in_range       = p.phase_out.length[filing_status]

# Manual inputs to test
qbi = np.array([80_000, 80_000, 80_000, 110_000, 110_000, 110_000, 200_000])
w2_wages = np.array([8_000, 8_000, 8_000, 60_000, 60_000, 60_000, 150_000])
ubia_property = np.array([50_000, 50_000, 50_000, 6_000, 6_000, 6_000, 0])
taxable_income = np.array([400_100, 460_100, 330_100, 400_100, 460_100, 330_100, 120_000])
threshold = np.repeat(340_100, qbi.shape[0])
phase_in_range = np.repeat(100_000, qbi.shape[0])
is_sstb = np.array([0, 0, 0, 1, 1, 1, 0])


# 1. Core inputs ---------------------------------------------------------
qbi = person("qualified_business_income", period)
w2_wages = person("w2_wages_from_qualified_business", period)
ubia_property = person("unadjusted_basis_qualified_property", period)
is_sstb = person("business_is_sstb", period)
reit_ptp_income = person("qualified_reit_and_ptp_income", period)

taxable_income = person.tax_unit("taxable_income_less_qbid", period)
filing_status = person.tax_unit("filing_status", period)

threshold = p.phase_out.start[filing_status]
phase_in_range = p.phase_out.length[filing_status]

# 2. 20 % of QBI ---------------------------------------------------------
qbi_twenty = p.max.rate * qbi

# 3. Wage / UBIA limitation ---------------------------------------------
wage_limit = p.max.w2_wages.rate * w2_wages  # 50 % of W‑2 wages
alt_limit = (
    p.max.w2_wages.alt_rate * w2_wages  # 25 % of W‑2 wages
    + p.max.business_property.rate * ubia_property  # 2.5 % of UBIA
)
wage_ubia_cap = max_(wage_limit, alt_limit)

# 4. Phase‑in percentage (§199A(b)(3)(B)) -------------------------------
over_threshold = max_(0, taxable_income - threshold)
phase_in_pct = min_(1, over_threshold / phase_in_range)

# 5. Applicable percentage for SSTBs ------------------------------------
applicable_pct = where(is_sstb, 1 - phase_in_pct, 1)

adj_qbi_twenty = qbi_twenty * applicable_pct
adj_cap = wage_ubia_cap * applicable_pct

limited_deduction = min_(adj_qbi_twenty, adj_cap)
excess = max_(0, adj_qbi_twenty - adj_cap)
phased_deduction = max_(0, adj_qbi_twenty - phase_in_pct * excess)

deduction_pre_cap = where(
    phase_in_pct == 0,
    adj_qbi_twenty,  # Below threshold: wage/UBIA limit does not apply.
    where(
        phase_in_pct < 1,
        max_(limited_deduction, phased_deduction),  # Inside phase‑in band.
        limited_deduction,  # Over the band: limitation fully applies.
    ),
)

# 6. REIT / PTP component (always 20 %) ---------------------------------
reit_ptp_deduction = p.max.rate * reit_ptp_income
total_before_income_cap = deduction_pre_cap + reit_ptp_deduction

# 7. Overall 20 % taxable‑income ceiling (§199A(a)(2)) ------------------
income_cap = p.max.rate * taxable_income
min_(total_before_income_cap, income_cap)
