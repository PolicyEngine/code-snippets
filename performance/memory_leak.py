"""
Quick memory diagnostic to test PolicyEngine calculate() calls in a loop.
"""
import gc
import psutil
import numpy as np
from policyengine_us import Microsimulation

FIX_THE_PROBLEM=False

print("Loading microsimulation...")
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

process = psutil.Process()
print(f"Initial memory: {process.memory_info().rss / 1024**3:.2f} GB")

print("\nTesting memory usage for 5 years...")
for year in range(2025, 2028):

    if FIX_THE_PROBLEM:
        sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")
    income_tax = sim.calculate("income_tax", period=year, map_to="household")
    baseline_weights = income_tax.weights.values
    values = income_tax.values

    total = np.sum(values * baseline_weights)

    gc.collect()
    mem_gb = process.memory_info().rss / 1024**3

    print(f"{year}: Total income tax = ${total/1e9:.1f}B, Memory = {mem_gb:.2f} GB")

print("\nIf memory keeps growing significantly, we have a memory leak problem.")
