# CPS ASEC Microdata Analysis

This folder contains scripts and documentation for downloading and analyzing CPS ASEC microdata directly from the Census Bureau.

## Quick Start

### 1. Download the Data

Download CPS ASEC data for your year of interest. The data is available from Census at:
https://www2.census.gov/programs-surveys/cps/datasets/

Example for 2023 data (released March 2024):

```bash
curl -o asecpub24csv.zip \
  https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip
unzip asecpub24csv.zip
```

Available years and URLs:
- 2024 data: `https://www2.census.gov/programs-surveys/cps/datasets/2025/march/asecpub25csv.zip`
- 2023 data: `https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub24csv.zip`
- 2022 data: `https://www2.census.gov/programs-surveys/cps/datasets/2023/march/asecpub23csv.zip`
- 2021 data: `https://www2.census.gov/programs-surveys/cps/datasets/2022/march/asecpub22csv.zip`
- 2020 data: `https://www2.census.gov/programs-surveys/cps/datasets/2021/march/asecpub21csv.zip`
- 2019 data: `https://www2.census.gov/programs-surveys/cps/datasets/2020/march/asecpub20csv.zip`
- 2018 data: `https://www2.census.gov/programs-surveys/cps/datasets/2019/march/asecpub19csv.zip`

### 2. File Contents

The ZIP contains three CSV files:

- **pppubXX.csv** - Person-level records (main file)
  - Contains individual person records
  - Includes SPM variables like `SNAPSUB`, `RESOURCES`, `POVTHRESHOLD`
  - Use `SPM_ID` to group to SPM unit level

- **ffpubXX.csv** - Family-level records
  - Contains family group data

- **hhpubXX.csv** - Household-level records
  - Contains household data

### 3. Key Variables

**SNAP-related:**
- `SNAPSUB` - SNAP benefits reported
- `SNAPLIM` - SNAP eligibility threshold (varies by year)

**SPM Variables:**
- `RESOURCES` - Total SPM resources
- `POVTHRESHOLD` - SPM poverty threshold
- `EQUIVSCALE` - SPM equivalence scale
- `POOR` - Indicator for SPM poverty status

**Other Important:**
- `SPM_ID` - SPM unit identifier (group persons by this to get unit-level data)
- `WEIGHT` or `A_FNLWGT` - Survey weight for population estimates
- `HAGE` - Age of household head
- `HRACE` - Race of household head
- `HHISP` - Hispanic indicator

### 4. Documentation

- Census CPS ASEC documentation: https://www2.census.gov/programs-surveys/cps/techdocs/
- Look for the annual ASEC codebook for variable definitions and codes
- SPM methodology is documented in Census research papers

### 5. Running Analysis

Use `starter_script.py` to begin analyzing the data. See the script for examples of:
- Loading specific columns
- Aggregating to SPM unit level
- Computing summary statistics
- Filtering subpopulations
