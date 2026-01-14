# Querying policy_data.db

Quick reference for finding calibration targets and strata.

## Database Location

```
policyengine_us_data/storage/calibration/policy_data.db
```

## Schema Overview

| Table | Purpose |
|-------|---------|
| `targets` | Calibration target values (e.g., person_count, adjusted_gross_income) |
| `strata` | Stratum definitions with unique hash |
| `stratum_constraints` | Filters defining each stratum (state, age, AGI bracket, etc.) |

### Key Columns

**targets**: `target_id`, `variable`, `period`, `stratum_id`, `value`, `active`

**stratum_constraints**: `stratum_id`, `constraint_variable`, `operation`, `value`

## Example: Minnesota (FIPS 27)

### Find all MN strata

```sql
SELECT * FROM stratum_constraints
WHERE constraint_variable = 'state_fips' AND value = '27';
```

### Find MN strata with AGI brackets

```sql
SELECT * FROM stratum_constraints
WHERE stratum_id IN (
  SELECT stratum_id FROM stratum_constraints
  WHERE constraint_variable = 'state_fips' AND value = '27'
  INTERSECT
  SELECT stratum_id FROM stratum_constraints
  WHERE constraint_variable = 'adjusted_gross_income'
)
ORDER BY stratum_id;
```

### Get person_count targets for MN by AGI bracket

```sql
SELECT * FROM targets
WHERE variable = 'person_count'
AND stratum_id IN (
  SELECT stratum_id FROM stratum_constraints
  WHERE constraint_variable = 'state_fips' AND value = '27'
  INTERSECT
  SELECT stratum_id FROM stratum_constraints
  WHERE constraint_variable = 'adjusted_gross_income'
);
```

### Get total AGI target for MN filers

```sql
SELECT * FROM targets
WHERE variable = 'adjusted_gross_income'
AND stratum_id IN (
  SELECT stratum_id FROM stratum_constraints
  WHERE constraint_variable = 'state_fips' AND value = '27'
);
```

## Useful Discovery Queries

### List all constraint variables

```sql
SELECT DISTINCT constraint_variable FROM stratum_constraints;
```

### List all target variables

```sql
SELECT DISTINCT variable FROM targets;
```

### View full stratum definition

```sql
SELECT * FROM stratum_constraints WHERE stratum_id = 10276;
```

## State FIPS Codes

Common codes: AL=1, AK=2, AZ=4, CA=6, CO=8, FL=12, IL=17, MN=27, NY=36, TX=48
