from policyengine_us import Microsimulation

sim_orig = Microsimulation()
sim_hack = Microsimulation(dataset="hf://policyengine/test/aca/enhanced_cps_2024_override.h5")

for label, sim, year in [("ORIGINAL 2024", sim_orig, 2024), ("HACKED 2025", sim_hack, 2025)]:
    eligible = sim.calculate("is_aca_ptc_eligible", period=year)
    has_ptc = sim.calculate("aca_ptc", period=year, map_to="person") > 0
    eligible_persons = float(eligible.sum())
    takeup_persons = float((has_ptc * eligible).sum())
    print(f"{label}:")
    print(f"  Eligible persons: {eligible_persons:,.0f}")
    print(f"  Persons with ACA PTC: {takeup_persons:,.0f}")
    print(f"  Takeup rate (persons): {takeup_persons / eligible_persons:.4f}")
    print()
