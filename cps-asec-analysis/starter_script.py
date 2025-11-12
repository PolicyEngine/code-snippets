"""
Starter script for analyzing CPS ASEC microdata.

Download the data first:
  curl -o asecpub25csv.zip https://www2.census.gov/programs-surveys/cps/datasets/2024/march/asecpub25csv.zip
  unzip asecpub25csv.zip
"""

import pandas as pd
import numpy as np

# ============================================================================
# BASIC LOADING
# ============================================================================

pppub = pd.read_csv('pppub25.csv')
hhpub = pd.read_csv('hhpub25.csv') 
ffpub = pd.read_csv('ffpub25.csv')  # Family records

hhpub.HSUP_WGT.sum()  # ASEC Supplement Final Weight, divide by 100
ffpub.FSUP_WGT.sum()  # Householder or Reference Person weight  in the universe of families

w_spm = pppub.SPM_WEIGHT.values / 100
np.sum(w_spm) / 1E6

#SPM_SNAPSub
#SPM unit's Supplemental Nutrition Assistance Program (SNAP)
#subsidy
#5 1542 (00000:99999)
#Values: $0 to $99,999
#Universe: All Persons
# It's on the record type of Person

#You're observing a common pattern in CPS ASEC data! Even though SPM_SNAPSUB appears on person-level records, it's actually a SPM unit-level value that gets repeated for every person in that unit.
#Notice in your data dictionary that the variable is described as "SPM unit's Supplemental Nutrition Assistance Program (SNAP) subsidy" - not the individual person's subsidy.

pppub.loc[pppub.SPM_SNAPSUB > 0].SPM_ID

# Yep, it's a person record
pppub.loc[pppub.SPM_ID == 24001][["PRECORD", "SPM_ID", "PPPOS", "SPM_WEIGHT", "SPM_HEAD", "SPM_FAMTYPE", "SPM_SNAPSUB"]]

# ASEC Supplement final weight, with universe as all persons
w_p = pppub.MARSUPWT / 100  # from the "two implied decimals"

spm_df = pppub.loc[pppub.SPM_HEAD==1]

np.sum(spm_df.SPM_SNAPSUB.values * (spm_df.SPM_WEIGHT.values / 100)) / 1E9

df = pd.read_csv('pppub25.csv', usecols=[
    'SPM_WEIGHT',
    'SPM_SNAPSUB',
    'SPM_HHISP',
    'SPM_POVTHRESHOLD',
    'SPM_HAGE',
    'SPM_HRACE',
    'SPM_RESOURCES'
])

df.SPM_WEIGHT.sum()


df.SPM_WEIGHT.sum()

#FSUP_WGT
#Householder or Reference Person weight

print(f"Loaded {len(df):,} person records")
print(f"Number of SPM units: {df['SPM_ID'].nunique():,}")

# ============================================================================
# AGGREGATE TO SPM UNIT LEVEL
# ============================================================================

# Get first person record per SPM unit (SPM variables are unit-level)
spm_units = df.groupby("SPM_ID").first().reset_index()

print(f"\nSPM unit level data: {len(spm_units):,} units")


# ============================================================================
# SNAP ANALYSIS
# ============================================================================

# Basic SNAP statistics
print(f"\n--- SNAP Statistics ---")
print(f"Units with SNAP: {(spm_units['SNAPSUB'] > 0).sum():,}")
print(f"Units without SNAP: {(spm_units['SNAPSUB'] == 0).sum():,}")
print(f"Average SNAP (all): ${spm_units['SNAPSUB'].mean():.2f}")
print(f"Average SNAP (recipients): ${spm_units[spm_units['SNAPSUB'] > 0]['SNAPSUB'].mean():.2f}")
print(f"Total SNAP (weighted): ${(spm_units['SNAPSUB'] * spm_units['WEIGHT']).sum():,.0f}")


# ============================================================================
# SPM POVERTY ANALYSIS
# ============================================================================

# Identify SPM poor (resources < poverty threshold)
spm_units['spm_poor'] = spm_units['RESOURCES'] < spm_units['POVTHRESHOLD']

print(f"\n--- SPM Poverty ---")
print(f"SPM poor: {spm_units['spm_poor'].sum():,} units")
print(f"SPM poverty rate: {spm_units['spm_poor'].mean():.2%}")

# Weighted poverty rate
weighted_poor = (spm_units[spm_units['spm_poor']]['WEIGHT'].sum() /
                 spm_units['WEIGHT'].sum())
print(f"Weighted SPM poverty rate: {weighted_poor:.2%}")


# ============================================================================
# SNAP BY POVERTY STATUS
# ============================================================================

print(f"\n--- SNAP Coverage ---")
snap_recipients = spm_units['SNAPSUB'] > 0
snap_poor = snap_recipients & spm_units['spm_poor']
snap_non_poor = snap_recipients & ~spm_units['spm_poor']

print(f"SNAP recipients who are SPM poor: {snap_poor.sum():,}")
print(f"SNAP recipients who are not SPM poor: {snap_non_poor.sum():,}")
print(f"Coverage rate (SNAP / SPM poor): {snap_poor.sum() / spm_units['spm_poor'].sum():.2%}")


# ============================================================================
# SUBGROUP ANALYSIS
# ============================================================================

# Example: Analysis by race
print(f"\n--- Analysis by Race ---")
for race in spm_units['HRACE'].unique():
    subset = spm_units[spm_units['HRACE'] == race]
    snap_pct = (subset['SNAPSUB'] > 0).mean()
    poverty_pct = subset['spm_poor'].mean()
    print(f"Race {race}: SNAP={snap_pct:.1%}, SPM Poverty={poverty_pct:.1%}")


# ============================================================================
# EXAMPLE: CUSTOM ANALYSIS
# ============================================================================

# Add your own analysis below:

# Example 1: Distribution of SNAP amounts among recipients
snap_recipients_df = spm_units[spm_units['SNAPSUB'] > 0]
print(f"\n--- SNAP Distribution (recipients only) ---")
print(snap_recipients_df['SNAPSUB'].describe())

# Example 2: Compare resource levels for poor vs non-poor
print(f"\n--- Resources by Poverty Status ---")
print(f"Poor - mean resources: ${spm_units[spm_units['spm_poor']]['RESOURCES'].mean():.2f}")
print(f"Non-poor - mean resources: ${spm_units[~spm_units['spm_poor']]['RESOURCES'].mean():.2f}")
