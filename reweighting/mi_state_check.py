"""Quick check of MI state H5: poverty by filer status + SNAP match.

Run:
    source ~/envs/temp/bin/activate
    python mi_state_check.py
"""

from policyengine_us import Microsimulation

MI_SNAP_TARGET = 3_061_361_572

sim = Microsimulation(dataset="hf://policyengine/test/mar/MI.h5")

# ── Section 1: Poverty by filer status ───────────────────────────────

poverty = sim.calculate("person_in_poverty", map_to="person").values
weights = sim.calculate("household_weight", map_to="person").values
is_filer = sim.calculate("tax_unit_is_filer", map_to="person").values.astype(bool)

total_pop = weights.sum()
overall = (poverty * weights).sum() / total_pop

filer_pop = weights[is_filer].sum()
filer_rate = (poverty[is_filer] * weights[is_filer]).sum() / filer_pop

nf_pop = weights[~is_filer].sum()
nf_rate = (poverty[~is_filer] * weights[~is_filer]).sum() / nf_pop

print("=" * 50)
print("Poverty by filer status")
print("=" * 50)
print(f"Overall:    {overall:.1%}  (pop {total_pop:,.0f})")
print(f"Filer:      {filer_rate:.1%}  (pop {filer_pop:,.0f})")
print(f"Non-filer:  {nf_rate:.1%}  (pop {nf_pop:,.0f})")

# ── Section 2: SNAP match ────────────────────────────────────────────

hw = sim.calculate("household_weight", map_to="household").values
snap_computed = (sim.calculate("snap", map_to="household").values * hw).sum()

print(f"\n{'=' * 50}")
print("SNAP totals")
print("=" * 50)
print(f"snap (computed):  ${snap_computed:,.0f}")
print(f"USDA target:      ${MI_SNAP_TARGET:,.0f}")
print(f"% error vs target: {(snap_computed - MI_SNAP_TARGET) / MI_SNAP_TARGET:+.1%}")

# It's interesting that snap is low but the overall blended poverty is so high
#
#● They're actually consistent — low SNAP and high poverty reinforce each other since snap flows into
#  spm_unit_net_income. If SNAP hit the $3.06B target, that extra ~$570M would lift some households above the SPM
#  threshold.
#
#  But the dominant factor is the non-filer composition. Non-filers are only 14% of the population yet contribute
#  ~6.5pp to the 17.7% rate (vs ~11.1pp from the 86% who are filers). The 45.8% non-filer poverty rate is the real
#  lever — closing the SNAP gap alone wouldn't dent it much since those non-filer households likely have very low
#  income across the board, not just low SNAP.
