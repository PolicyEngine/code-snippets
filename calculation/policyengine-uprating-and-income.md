# PolicyEngine US: Uprating System and Income Variables

A comprehensive guide to understanding how PolicyEngine US calculates federal income tax and projects values forward in time.

---

## Table of Contents
1. [Federal Income Tax Dependency Tree](#federal-income-tax-dependency-tree)
2. [IRS Gross Income Sources](#irs-gross-income-sources)
3. [Uprating System Architecture](#uprating-system-architecture)
4. [Two-Tier Uprating System](#two-tier-uprating-system)
5. [Practical Examples](#practical-examples)
6. [Key Insights](#key-insights)

---

## Federal Income Tax Dependency Tree

### Top-Level Formula

**File:** `/policyengine_us/variables/gov/irs/tax/federal_income/income_tax.py:12-24`

```python
income_tax = income_tax_before_refundable_credits - income_tax_refundable_credits
```

### Complete Dependency Map

```
income_tax
│
├─[PARAMETER CHECK] abolish_federal_income_tax → if True, return 0
│
└─[FORMULA] = income_tax_before_refundable_credits - income_tax_refundable_credits
   │
   ├─── income_tax_before_refundable_credits
   │    │
   │    ├─[ADD] income_tax_before_credits
   │    │      │
   │    │      ├── income_tax_main_rates
   │    │      │   ├── taxable_income
   │    │      │   │   ├── adjusted_gross_income (AGI)
   │    │      │   │   │   ├── irs_gross_income
   │    │      │   │   │   └── above_the_line_deductions
   │    │      │   │   ├── exemptions
   │    │      │   │   └── taxable_income_deductions (standard/itemized)
   │    │      │   ├── capital_gains_excluded_from_taxable_income
   │    │      │   └── filing_status
   │    │      │
   │    │      ├── capital_gains_tax
   │    │      │   ├── net_capital_gain
   │    │      │   ├── adjusted_net_capital_gain
   │    │      │   ├── qualified_dividend_income
   │    │      │   ├── unrecaptured_section_1250_gain
   │    │      │   ├── capital_gains_28_percent_rate_gain
   │    │      │   ├── taxable_income
   │    │      │   └── filing_status
   │    │      │
   │    │      └── alternative_minimum_tax
   │    │          ├── amt_base_tax
   │    │          ├── amt_tax_including_cg
   │    │          ├── foreign_tax_credit_potential
   │    │          ├── regular_tax_before_credits
   │    │          └── capital_gains_tax
   │    │
   │    ├─[ADD] net_investment_income_tax
   │    │      ├── adjusted_gross_income
   │    │      ├── filing_status
   │    │      └── net_investment_income
   │    │          ├── taxable_interest_income
   │    │          ├── dividend_income
   │    │          ├── rental_income
   │    │          └── loss_limited_net_capital_gains
   │    │
   │    ├─[ADD] recapture_of_investment_credit (stub - always 0)
   │    ├─[ADD] unreported_payroll_tax (stub - always 0)
   │    ├─[ADD] qualified_retirement_penalty (stub - always 0)
   │    │
   │    └─[SUBTRACT] income_tax_capped_non_refundable_credits
   │               │
   │               └── [MIN of available credits vs. tax liability]
   │                   ├── foreign_tax_credit
   │                   ├── cdcc (child/dependent care)
   │                   ├── non_refundable_american_opportunity_credit
   │                   ├── lifetime_learning_credit
   │                   ├── savers_credit
   │                   ├── residential_clean_energy_credit
   │                   ├── energy_efficient_home_improvement_credit
   │                   ├── elderly_disabled_credit
   │                   ├── new_clean_vehicle_credit
   │                   ├── used_clean_vehicle_credit
   │                   └── non_refundable_ctc
   │
   └─── income_tax_refundable_credits
        │
        ├── eitc (Earned Income Tax Credit)
        │   ├── takes_up_eitc
        │   ├── eitc_maximum
        │   ├── eitc_phased_in
        │   └── eitc_reduction
        │
        ├── refundable_american_opportunity_credit
        │
        ├── refundable_ctc (Child Tax Credit)
        │   ├── ctc_refundable_maximum
        │   ├── ctc_phase_out
        │   ├── ctc_phase_in
        │   └── ctc_limiting_tax_liability
        │
        ├── recovery_rebate_credit
        └── refundable_payroll_tax_credit
```

### Key Points

1. **Main Formula**: Tax = (Tax before refundable credits) - (Refundable credits)

2. **Tax Before Refundable Credits** includes:
   - Regular income tax (on ordinary income)
   - Capital gains tax (preferential rates)
   - AMT (Alternative Minimum Tax)
   - Net Investment Income Tax (3.8% surtax)
   - MINUS non-refundable credits (capped at tax liability)

3. **Refundable Credits** can exceed tax liability and create a negative income tax

4. **Core Building Block**: `adjusted_gross_income` (AGI) is the foundation for most calculations

5. **Filing Status**: Affects brackets, thresholds, and credit amounts throughout

---

## IRS Gross Income Sources

### Top-Level Structure

**File:** `/policyengine_us/variables/gov/irs/income/taxable_income/adjusted_gross_income/irs_gross_income/irs_gross_income.py:13-20`

```python
def formula(person, period, parameters):
    sources = parameters(period).gov.irs.gross_income.sources
    total = 0
    not_dependent = ~person("is_tax_unit_dependent", period)
    for source in sources:
        # Add positive values only - losses are deducted later.
        total += not_dependent * max_(0, add(person, period, [source]))
    return total
```

### 17 Income Sources

**Parameter File:** `/policyengine_us/parameters/gov/irs/gross_income/sources.yaml`

```
irs_gross_income (only non-dependents, positive values only)
│
└─[SUM OF 17 INCOME SOURCES]
   │
   ├── 1. irs_employment_income
   │      ├── employment_income
   │      │   ├── employment_income_before_lsr (INPUT)
   │      │   └── employment_income_behavioral_response
   │      │       └── labor_supply_behavioral_response
   │      │           ├── income_elasticity_lsr
   │      │           └── substitution_elasticity_lsr
   │      └── pre_tax_contributions (subtracted)
   │          ├── traditional_401k_contributions (INPUT)
   │          ├── traditional_403b_contributions (INPUT)
   │          ├── health_insurance_premiums (INPUT)
   │          └── health_savings_account_payroll_contributions (INPUT)
   │
   ├── 2. self_employment_income
   │      ├── self_employment_income_before_lsr (INPUT)
   │      └── self_employment_income_behavioral_response
   │          └── labor_supply_behavioral_response
   │
   ├── 3. partnership_s_corp_income (INPUT)
   │
   ├── 4. farm_income (INPUT)
   │
   ├── 5. farm_rent_income (INPUT)
   │
   ├── 6. capital_gains
   │      ├── short_term_capital_gains (INPUT)
   │      └── long_term_capital_gains
   │          ├── long_term_capital_gains_before_response (INPUT)
   │          └── capital_gains_behavioral_response
   │              ├── relative_capital_gains_mtr_change
   │              └── capital_gains_elasticity (PARAMETER)
   │
   ├── 7. taxable_interest_income (INPUT)
   │
   ├── 8. rental_income (INPUT)
   │
   ├── 9. dividend_income
   │      ├── qualified_dividend_income (INPUT)
   │      └── non_qualified_dividend_income (INPUT)
   │
   ├── 10. taxable_pension_income
   │       ├── taxable_public_pension_income (INPUT)
   │       └── taxable_private_pension_income (INPUT)
   │
   ├── 11. debt_relief (INPUT)
   │
   ├── 12. taxable_unemployment_compensation
   │       ├── is_tax_unit_head (to avoid double counting)
   │       └── tax_unit_taxable_unemployment_compensation
   │           ├── tax_unit_unemployment_compensation
   │           │   └── unemployment_compensation (INPUT)
   │           ├── taxable_uc_agi
   │           │   └── [AGI calculated WITHOUT taxable UC]
   │           └── filing_status
   │
   ├── 13. taxable_social_security
   │       ├── is_tax_unit_head / is_tax_unit_spouse
   │       ├── tax_unit_taxable_social_security
   │       │   ├── tax_unit_social_security
   │       │   │   └── social_security
   │       │   │       ├── social_security_retirement (INPUT)
   │       │   │       ├── social_security_disability (INPUT)
   │       │   │       ├── social_security_survivors (INPUT)
   │       │   │       └── social_security_dependents (INPUT)
   │       │   ├── tax_unit_combined_income_for_social_security_taxability
   │       │   │   ├── taxable_ss_magi
   │       │   │   │   └── [Modified AGI WITHOUT taxable SS/UC]
   │       │   │   └── tax_unit_social_security
   │       │   ├── filing_status
   │       │   └── cohabitating_spouses
   │       └── [proportional allocation to persons]
   │
   ├── 14. illicit_income (INPUT)
   │
   ├── 15. taxable_retirement_distributions
   │       ├── taxable_ira_distributions (INPUT)
   │       ├── taxable_401k_distributions (INPUT)
   │       ├── taxable_sep_distributions (INPUT)
   │       ├── taxable_403b_distributions (INPUT)
   │       └── keogh_distributions (INPUT)
   │
   ├── 16. miscellaneous_income (INPUT)
   │
   └── 17. ak_permanent_fund_dividend
           └── [parameter-driven, Alaska residents only]
```

### Input Variables (27 Total)

These are raw inputs from CPS or user entry:

**Employment & Business:**
- employment_income_before_lsr
- self_employment_income_before_lsr
- partnership_s_corp_income
- farm_income, farm_rent_income

**Capital Income:**
- short_term_capital_gains, long_term_capital_gains_before_response
- taxable_interest_income
- qualified_dividend_income, non_qualified_dividend_income
- rental_income

**Retirement:**
- taxable_public_pension_income, taxable_private_pension_income
- taxable_ira_distributions, taxable_401k_distributions
- taxable_sep_distributions, taxable_403b_distributions
- keogh_distributions

**Benefits:**
- unemployment_compensation
- social_security_retirement, social_security_disability
- social_security_survivors, social_security_dependents

**Pre-tax Deductions:**
- traditional_401k_contributions, traditional_403b_contributions
- health_insurance_premiums
- health_savings_account_payroll_contributions

**Other:**
- debt_relief, illicit_income, miscellaneous_income

### Behavioral Response Variables (3 Sources)

1. **Employment Income**: Responds to marginal tax rate changes via labor supply elasticity
2. **Self-Employment Income**: Responds oppositely to employment (substitution effect)
3. **Long-term Capital Gains**: Responds to capital gains tax rate changes via realization elasticity

### Circular Dependencies (2 Sources)

1. **Taxable Unemployment Compensation**:
   - Depends on AGI, which depends on gross income, which includes taxable UC
   - Resolved by calculating `taxable_uc_agi` WITHOUT taxable UC

2. **Taxable Social Security**:
   - Depends on modified AGI, which depends on gross income, which includes taxable SS
   - Resolved by calculating `taxable_ss_magi` WITHOUT taxable SS or taxable UC

### Important Filters

- **Non-dependents only**: Dependents' income goes to $0 in gross income calculation
- **Positive values only**: Uses `max_(0, value)` to exclude losses

---

## Uprating System Architecture

### Three Levels of Uprating

PolicyEngine uses a hierarchical three-level uprating system to project values forward in time.

#### Level 1: Economic Indices (Base Growth Rates)

**Purpose:** Provide fundamental growth rates for different sectors

**Examples:**
```
gov.irs.uprating          → Chained CPI-U (tax parameters)
gov.ssa.uprating          → CPI-W (Social Security)
gov.usda.snap.uprating    → Thrifty Food Plan
gov.bls.cpi.cpi_u         → Consumer Price Index
```

**Characteristics:**
- Have explicit values through ~2035 (from CBO projections or actual data)
- Extended to 2100 programmatically using constant growth rate
- Updated annually with new CBO forecasts

**File Location:** `/policyengine_us/parameters/gov/[agency]/uprating.yaml`

#### Level 2: Policy Parameters (Use Indices)

**Purpose:** Specific benefit amounts, thresholds, and brackets that follow economic indices

**Examples:**
```
SSI amounts               → uprating: gov.ssa.uprating
Tax brackets              → uprating: gov.irs.uprating
SNAP benefits             → uprating: gov.usda.snap.uprating
```

**Characteristics:**
- Have explicit values for recent years (actual legislation)
- Auto-calculate future years using Level 1 indices
- Can include rounding rules (e.g., round down to nearest $50)

**File Location:** `/policyengine_us/parameters/gov/[program]/[parameter].yaml`

#### Level 3: Variables (Point to Calibration Parameters)

**Purpose:** Calibrate microdata to match aggregate national totals

**Examples:**
```python
# In variable definition:
class employment_income(Variable):
    uprating = "calibration.gov.irs.soi.employment_income"
```

**Characteristics:**
- Used to scale individual microdata values
- Ensures microsimulation totals match IRS/SSA/Census aggregates
- Can chain to other uprating parameters

**File Location:** `/policyengine_us/variables/[category]/[variable].py`

### Implementation Details

**System Initialization** (`/policyengine_us/system.py:82-88`):

```python
# Order of operations during system initialization:
1. set_irs_uprating_parameter()      # Extend uprating through 2100
2. homogenize_parameter_structures() # Normalize parameter formats
3. propagate_parameter_metadata()    # Spread metadata (including uprating)
4. interpolate_parameters()          # Fill missing values
5. uprate_parameters()               # Apply uprating to parameter values
6. propagate_parameter_metadata()    # Again after uprating
```

**Key Insight:** Uprating happens at initialization, NOT at calculation time. When formulas execute, they receive already-uprated parameter values.

### How to Know Which Calculation Method is Used

**Question:** If a variable has both `adds` and `uprating`, which is used?

**Answer:** BOTH, but at different stages:

```python
# Example: employment_income
class employment_income(Variable):
    adds = ["employment_income_before_lsr", "employment_income_behavioral_response"]
    uprating = "calibration.gov.irs.soi.employment_income"
```

**Stage 1: Input Uprating (microdata calibration)**
- The base input `employment_income_before_lsr` gets uprated from data year to simulation year
- Scales individual values proportionally to match aggregate targets
- Example: 2024→2025 multiply by 1.0488 (based on CBO projections)

**Stage 2: Formula Calculation**
- Then the `adds` declaration executes:
  ```python
  employment_income = employment_income_before_lsr + employment_income_behavioral_response
  ```

**The Rule:** If a variable has `adds` or a `formula`, it ALWAYS calculates using that logic. The `uprating` attribute is for calibrating the underlying microdata, not replacing the calculation.

**Practical Test Results:**
```
2024 total: $9,979,443,414,627
2025 total: $10,564,079,334,278
Actual ratio: 1.0586
Target ratio: 1.0488
```
The totals are close but not exact because behavioral responses and other factors affect the final aggregate.

---

## Two-Tier Uprating System

### What is "Uprating has an Uprating"?

Parameters themselves can have `uprating` metadata that specifies how their values should be uprated over time. This creates a two-tier cascade.

### Example 1: SSI Benefits (Simple)

**Tier 1: The Index**

File: `/policyengine_us/parameters/gov/ssa/uprating.yaml`

```yaml
description: SSA uprating based on CPI-W in Q3 of prior year
values:
  2022-01-01: 268.421  # 2021Q3 CPI-W average
  2023-01-01: 291.901  # 2022Q3 CPI-W average
  2024-01-01: 301.236  # 2023Q3 CPI-W average
  2025-01-01: 308.767  # 2.5% COLA announced by SSA
  2026-01-01: 318.155  # CBO forecast
  # ... through 2035
  # 2036-2100: Extended programmatically
```

**Tier 2: The Benefit Amount**

File: `/policyengine_us/parameters/gov/ssa/ssi/amount/individual.yaml`

```yaml
description: Monthly maximum Federal SSI payment amounts
values:
  2023-01-01: 914
  2024-01-01: 943
  2025-01-01: 967
  # 2026+: NO explicit values
metadata:
  uprating: gov.ssa.uprating  # Points to Tier 1
  unit: currency-USD
```

**How it Works:**

1. **For years WITH explicit values (2023-2025):** Use value directly from YAML
   - 2025: $967/month (actual SSA value)

2. **For years WITHOUT explicit values (2026+):** Apply uprating formula
   ```python
   SSI_2026 = SSI_2025 × (uprating_2026 / uprating_2025)
   SSI_2026 = $967 × (318.155 / 308.767) = $996.40
   ```

3. **The index extends to 2100:**
   - 2022-2025: Actual CPI-W data
   - 2026-2035: CBO projections
   - 2036-2100: Constant growth rate from 2034→2035

**Verification:**
```
2025: $967.00 (explicit)
2026: $996.40 (calculated)
2027: $1,021.44 (calculated)
2050: $1,697.58 (calculated)
```

### Example 2: NY WFTC (With Rounding)

**Tier 2 Parameter with Rounding Rules:**

File: `/policyengine_us/parameters/gov/contrib/states/ny/wftc/amount/max.yaml`

```yaml
description: NY Working Families Tax Credit max amount per child
metadata:
  uprating:
    parameter: gov.irs.uprating  # Tier 1 reference
    rounding:
      type: downwards
      interval: 50    # Round down to nearest $50
values:
  2025-01-01: 550   # From legislation
  2026-01-01: 800
  2027-01-01: 1_000
  2028-01-01: 1_200
  2029-01-01: 1_600
  # 2030+: calculated via uprating
```

**Calculation for 2030:**

```
Step 1: Get base value
  2029 value: $1,600 (last explicit value)

Step 2: Apply uprating ratio
  IRS uprating 2029: 185.950
  IRS uprating 2030: 189.650
  Growth: 1.99%
  Unrounded: $1,600 × (189.650 / 185.950) = $1,631.84

Step 3: Apply rounding
  Round down to nearest $50: $1,600.00

Result: 2030 value = $1,600
```

By 2031, cumulative inflation pushes the value to $1,650.

**Verification:**
```
2029: $1,600 (explicit from legislation)
2030: $1,600 (uprated 1.99% = $1,631.84, rounded down)
2031: $1,650 (uprated 1.98% = $1,663.53, rounded down)
2035: $1,750 (continued uprating)
2040: $1,950 (continued uprating)
```

### Benefits of Two-Tier System

1. **Historical Accuracy:** Use actual values when known
2. **Automatic Projection:** System calculates future values
3. **Consistency:** All related parameters use same index
4. **Maintainability:** Update one index, all parameters update
5. **Flexibility:** Can add rounding rules, intervals, etc.

### The Complete Cascade

```
Level 3: Variable (microdata)
└── uprating: calibration.gov.irs.soi.employment_income
    │
    Level 2: Calibration Parameter
    └── uprating: calibration.gov.cbo.income_by_source.employment_income
        │
        Level 1: Economic Index
        └── CBO projections based on economic growth
```

---

## Practical Examples

### Example 1: Employment Income Uprating

**Variable Declaration:**

```python
# /policyengine_us/variables/input/employment_income.py
class employment_income(Variable):
    value_type = float
    entity = Person
    label = "employment income"
    unit = USD
    definition_period = YEAR
    adds = [
        "employment_income_before_lsr",
        "employment_income_behavioral_response",
    ]
    uprating = "calibration.gov.irs.soi.employment_income"
```

**Calibration Parameter:**

```yaml
# /policyengine_us/parameters/calibration/gov/irs/soi/employment_income.yaml
description: Total employment income
values:
  2015-01-01: 7_112_222_959_000
  2020-01-01: 8_416_495_535_000
  2021-01-01: 9_022_352_941_000
metadata:
  uprating: calibration.gov.cbo.income_by_source.employment_income
```

**CBO Projection:**

```yaml
# /policyengine_us/parameters/calibration/gov/cbo/income_by_source.yaml
employment_income:
  2021-01-01: 9_022_400_000_000
  2022-01-01: 9_739_000_000_000
  2023-01-01: 10_255_100_000_000
  2024-01-01: 10_858_700_000_000
  2025-01-01: 11_389_100_000_000
  # ... through 2035
```

**Calculation Flow:**

1. CPS microdata has person with $50,000 employment income in 2021
2. Simulate for 2025:
   - Get uprating ratio: 11,389,100 / 9,022,400 = 1.2623
   - Scale person's income: $50,000 × 1.2623 = $63,115
3. Calculate behavioral response (based on MTR changes)
4. Sum: employment_income = base + behavioral_response

### Example 2: Default Uprating for Inputs

**System Tool:** `/policyengine_us/tools/default_uprating.py`

```python
def add_default_uprating(system):
    """
    108 input variables get default AGI-based uprating
    if they don't already have an uprating parameter
    """
    for variable in system.variables.values():
        if (variable.name in INPUT_VARIABLES) and (variable.uprating is None):
            variable.uprating = (
                "calibration.gov.cbo.income_by_source.adjusted_gross_income"
            )
```

This ensures all income sources grow consistently with the economy even if not explicitly specified.

### Example 3: Checking Uprating in Practice

```python
from policyengine_us import Microsimulation

sim = Microsimulation()

# Calculate for different years
emp_2024 = sim.calculate('employment_income', 2024)
emp_2025 = sim.calculate('employment_income', 2025)

# Check totals
total_2024 = emp_2024.sum()  # $9,979,443,414,627
total_2025 = emp_2025.sum()  # $10,564,079,334,278
ratio = total_2025 / total_2024  # 1.0586

# Check target
params = sim.tax_benefit_system.parameters
target_2024 = params.calibration.gov.irs.soi.employment_income('2024-01-01')
target_2025 = params.calibration.gov.irs.soi.employment_income('2025-01-01')
target_ratio = target_2025 / target_2024  # 1.0488

# Close but not exact due to behavioral responses
```

### Example 4: Uprating Extension to 2100

**File:** `/policyengine_us/parameters/uprating_extensions.py`

```python
def extend_parameter_values(
    parameter: Parameter,
    last_projected_year: int,
    end_year: int = 2100,
) -> None:
    """
    Extend a parameter's values using the growth rate
    from the last two years of projections.
    """
    # Calculate growth rate from last two years
    second_to_last = parameter(f"{last_projected_year - 1}-01-01")
    last = parameter(f"{last_projected_year}-01-01")
    growth_rate = last / second_to_last

    # Apply growth rate for years beyond projections
    for year in range(last_projected_year + 1, end_year + 1):
        previous = parameter(f"{year - 1}-01-01")
        new = previous * growth_rate
        parameter.update(period=f"year:{year}-01-01:1", value=new)
```

**Applied to:**
- IRS uprating (Chained CPI-U calculation)
- SNAP uprating (October values, through 2100)
- SSA uprating (January values, through 2100)
- HHS uprating (January values, through 2100)
- CPI-U, Chained CPI-U, CPI-W (February values, through 2100)

### Example 5: Common Uprating Parameters

**From Variable Declarations:**

```python
# Employment income
uprating = "calibration.gov.irs.soi.employment_income"

# Self-employment income
uprating = "calibration.gov.irs.soi.self_employment_income"

# CPI-indexed values (rent, child support, etc.)
uprating = "gov.bls.cpi.cpi_u"

# Healthcare expenses
uprating = "calibration.gov.hhs.cms.moop_per_capita"

# Population weights
uprating = "calibration.gov.census.populations.total"
```

---

## Key Insights

### 1. Uprating Happens at Initialization, Not Runtime

When `uprate_parameters()` is called during system initialization:
- All parameter values are adjusted for the target year
- Variables receive already-uprated values when they call `parameters(period)`
- No runtime overhead from uprating calculations

### 2. Variables with `adds` Always Use Formula Logic

Even if a variable has both `adds` and `uprating`:
- The formula/adds ALWAYS executes
- Uprating is for calibrating the underlying microdata
- Uprating doesn't replace calculation logic

### 3. Two-Tier Uprating Enables Smart Projection

**Historical Years:** Use actual legislative values
**Near-term Future:** Use CBO economic projections
**Long-term Future:** Extrapolate using constant growth rate

This balances accuracy (when we know values) with consistency (when we project).

### 4. Rounding Rules Matter for Policy Accuracy

Many benefit programs round to specific intervals:
- Tax brackets: Round to nearest $25 or $50
- State benefits: Round to nearest $10, $50, or $100
- Including rounding ensures projections match real policy

### 5. Circular Dependencies are Handled Elegantly

Taxable unemployment and Social Security both depend on AGI, which includes them:
- System calculates modified AGI WITHOUT the circular component
- Applies taxability rules using modified AGI
- Includes result in full AGI

### 6. Behavioral Responses Add Realism

Three income sources have behavioral elasticities:
- **Employment income:** Responds to labor income MTR changes
- **Self-employment income:** Opposite response (substitution)
- **Capital gains:** Responds to capital gains tax rate changes

This captures how people adjust to policy changes.

### 7. Default Uprating Prevents Gaps

108 input variables automatically get AGI-based uprating if not specified:
- Ensures all income grows with the economy
- Maintains relative income distribution
- Prevents artificial changes from static values

### 8. Aggregate Calibration Ensures Accuracy

Variable uprating calibrates microdata to match:
- IRS SOI totals (actual tax data)
- CBO projections (economic forecasts)
- Census data (population totals)

This grounds PolicyEngine's microsimulation in empirical reality.

### 9. Three-Level Architecture is Maintainable

```
Economic Index (CPI, wages, etc.)
    ↓
Policy Parameter (SSI amount, tax bracket)
    ↓
Variable (person-level income)
```

Update one economic index, all dependent parameters and variables update automatically.

### 10. Extension to 2100 Enables Long-Term Analysis

All uprating factors extend through 2100:
- Enables analysis of far-future policy changes
- Uses conservative assumptions (constant growth rate)
- Maintains internal consistency across all parameters

---

## File Reference Index

### Key System Files

- `/policyengine_us/system.py` - System initialization and uprating orchestration
- `/policyengine_us/tools/default_uprating.py` - Default uprating for input variables
- `/policyengine_us/parameters/uprating_extensions.py` - Extension to 2100

### Income Tax Variables

- `/policyengine_us/variables/gov/irs/tax/federal_income/income_tax.py`
- `/policyengine_us/variables/gov/irs/tax/federal_income/income_tax_before_refundable_credits.py`
- `/policyengine_us/variables/gov/irs/credits/income_tax_refundable_credits.py`

### Gross Income Variables

- `/policyengine_us/variables/gov/irs/income/taxable_income/adjusted_gross_income/irs_gross_income/irs_gross_income.py`
- `/policyengine_us/variables/gov/irs/income/taxable_income/adjusted_gross_income/irs_gross_income/irs_employment_income.py`
- `/policyengine_us/variables/input/employment_income.py`
- `/policyengine_us/variables/input/employment_income_before_lsr.py`

### Uprating Parameters

- `/policyengine_us/parameters/gov/ssa/uprating.yaml` - Social Security COLA
- `/policyengine_us/parameters/gov/irs/uprating.yaml` - IRS Chained CPI-U
- `/policyengine_us/parameters/gov/usda/snap/uprating.yaml` - SNAP Thrifty Food Plan
- `/policyengine_us/parameters/gov/bls/cpi/cpi_u.yaml` - CPI-U
- `/policyengine_us/parameters/calibration/gov/cbo/income_by_source.yaml` - CBO projections

### Test Files

- `/policyengine_us/tests/policy/baseline/parameters/test_uprating_extensions.py`

---

## Conclusion

PolicyEngine's uprating system is a sophisticated multi-tiered architecture that:

1. **Grounds simulations in reality** through aggregate calibration
2. **Projects consistently** using economic indices and CBO forecasts
3. **Maintains accuracy** by using actual values when known
4. **Enables long-term analysis** through extrapolation to 2100
5. **Stays maintainable** through hierarchical parameter structure

Understanding this system is crucial for:
- Interpreting PolicyEngine results
- Adding new variables or parameters
- Debugging unexpected behavior
- Contributing to the codebase

The three-level architecture (indices → parameters → variables) combined with two-tier cascading (parameters with their own uprating) creates a flexible, accurate, and maintainable system for economic policy microsimulation.

---

**Document Version:** 1.0
**Last Updated:** 2025-10-23
**Author:** Analysis of PolicyEngine US codebase
