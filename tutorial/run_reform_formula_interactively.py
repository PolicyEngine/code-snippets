# Make sure python instance is instantiated from -data repo, if using editable install
from policyengine_us_data.datasets.cps.enhanced_cps import EnhancedCPS_2024


from policyengine_core.periods import period as period_  # alias, "period" will be a var
from policyengine_core.populations.population import Population
from policyengine_core.entities import Entity
from policyengine_us import Microsimulation


# Create a simulation as we will harvest its internals
sim = Microsimulation(dataset=EnhancedCPS_2024)

period = period_('2024')
print(period)

# Entity init arguments are: the key, the plural, the label, the doc string
Person = Entity("person", "people", "Person", "A person")
print(Person.key)

# Population: Lowercase p person
person = Population(Person)
print(f"The person population is a population of {person.entity.plural}")
person.simulation = sim  # populations need simulations
person.entity.set_tax_benefit_system(sim.tax_benefit_system)  # and a tax benefit system through their entity

parameters = sim.tax_benefit_system.parameters


# ---- and NOW you can start to run the formula in: 
# /policyengine_us/reforms/reconciliation/reconciled_qbid.py  ------- 

# Inside def formula(person, period, parameters):
p = parameters(period).gov.irs.deductions
qbi = person("qualified_business_income", period)
qbid_max = p.qbi.max.rate * qbi
