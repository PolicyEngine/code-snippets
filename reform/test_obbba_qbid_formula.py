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
# Population: Lowercase p person
person = Population(Person)
print(f"The person population is a population of {person.entity.plural}")
person.simulation = sim  # populations need simulations
person.entity.set_tax_benefit_system(sim.tax_benefit_system)  # and a tax benefit system through their entity

parameters = sim.tax_benefit_system.parameters

p = parameters(period).gov.irs.deductions.qbi
p_ref = parameters(period).gov.contrib.reconciliation.qbid

# User inputs
qbi = np.array([80_000, 80_000, 80_000, 110_000, 110_000, 110_000, 200_000])
w2_wages = np.array([8_000, 8_000, 8_000, 60_000, 60_000, 60_000, 150_000])
ubia_property = np.array([50_000, 50_000, 50_000, 6_000, 6_000, 6_000, 0])
taxable_income = np.array([400_100, 460_100, 330_100, 400_100, 460_100, 330_100, 120_000])
threshold = np.repeat(340_100, qbi.shape[0])
phase_in_range = np.repeat(100_000, qbi.shape[0])
is_sstb = np.array([0, 0, 0, 1, 1, 1, 0])
reit_ptp_income  = np.repeat(0, qbi.shape[0])
bdc_income  = np.repeat(0, qbi.shape[0])

# ------------------------------------------------------------------
# 1. Core inputs ----------------------------------------------------
# ------------------------------------------------------------------
qbi = person("qualified_business_income", period)
is_sstb = person("business_is_sstb", period)

reit_ptp_income = person("qualified_reit_and_ptp_income", period)
bdc_income = person("qualified_bdc_income", period)

taxable_income = person.tax_unit("taxable_income_less_qbid", period)
filing_status = person.tax_unit("filing_status", period)

#threshold = p.phase_out.start[filing_status]          # §199A(e)(2)
phase_in_rate = 0.75 #p_ref.phase_out_rate                  # 75 % “phase-in” rate

# ------------------------------------------------------------------
# 2. 23 % of total QBI ---------------------------------------------
# ------------------------------------------------------------------
qbi_twenty_three =.23 * qbi  # TODO: make sure this is set right

# ------------------------------------------------------------------
# 3. Wage / UBIA limitation (non-SSTB only) ------------------------
# ------------------------------------------------------------------
w2_wages = person("w2_wages_from_qualified_business", period)
ubia_property = person("unadjusted_basis_qualified_property", period)

qbi_non_sstb = where(is_sstb, 0, qbi)
w2_wages_non_sstb = where(is_sstb, 0, w2_wages)
ubia_property_non_sstb = where(is_sstb, 0, ubia_property)

wage_limit = p.max.w2_wages.rate * w2_wages_non_sstb          # 50 % wages
alt_limit = (
    p.max.w2_wages.alt_rate * w2_wages_non_sstb               # 25 % wages
    + p.max.business_property.rate * ubia_property_non_sstb   # 2.5 % UBIA
)
wage_ubia_cap = max_(wage_limit, alt_limit)

step1_uncapped = p.max.rate * qbi_non_sstb
step1_deduction = min_(step1_uncapped, wage_ubia_cap)

# ------------------------------------------------------------------
# 4. Limitation phase-in amount (75 % × excess) --------------------
# ------------------------------------------------------------------
excess_income = max_(0, taxable_income - threshold)
phase_in_amount = phase_in_rate * excess_income

step2_deduction = max_(0, qbi_twenty_three - phase_in_amount)

# ------------------------------------------------------------------
# 5. QBI component: greater of Step 1 or Step 2 --------------------
# ------------------------------------------------------------------
qbi_component = max_(step1_deduction, step2_deduction)

# ------------------------------------------------------------------
# 6. REIT, PTP, and optional BDC component (always 23 %) -----------
# ------------------------------------------------------------------
reit_ptp_bdc_base = reit_ptp_income + where(
    #p_ref.use_bdc_income,
    False,
    bdc_income,
    0,
)
reit_ptp_bdc_deduction = p.max.rate * reit_ptp_bdc_base

total_before_income_cap = qbi_component + reit_ptp_bdc_deduction

# ------------------------------------------------------------------
# 7. Overall 23 % taxable-income ceiling (§199A(a)(2)) -------------
# ------------------------------------------------------------------
income_cap = p.max.rate * taxable_income

min_(total_before_income_cap, income_cap)
