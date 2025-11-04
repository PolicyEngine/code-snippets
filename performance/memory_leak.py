from pathlib import Path
import gc
import psutil

import numpy as np

from policyengine_us import Microsimulation

BRAND_NEW_SIM=False
CLEAR_THE_CACHE=True

print("Loading microsimulation...")
sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

process = psutil.Process()
print(f"Initial memory: {process.memory_info().rss / 1024**3:.2f} GB")

print("\nTesting memory usage for 5 years...")
for year in range(2025, 2028):

    if BRAND_NEW_SIM:
        sim = Microsimulation(dataset="hf://policyengine/policyengine-us-data/enhanced_cps_2024.h5")

    if CLEAR_THE_CACHE: 

        # Delete the actual cache folder
        data_dir = Path(sim.data_storage_dir)
        for cache_folder in data_dir.glob("*_variable_cache"):
            shutil.rmtree(cache_folder)
        
        # Clear in-memory caches
        for computed_variable in sim.tax_benefit_system.variables:
            if computed_variable not in sim.input_variables:
                sim.delete_arrays(computed_variable)
        
        # Check what's in the on-disk storage
        for holder in sim.populations['person']._holders.values():
            if holder._disk_storage:
                print(f"{holder.variable.name}: has disk storage at {holder._disk_storage.storage_dir}")
                holder._disk_storage.delete()  # Clear it

    income_tax = sim.calculate("income_tax", period=year, map_to="household")
    baseline_weights = income_tax.weights.values
    values = income_tax.values

    total = np.sum(values * baseline_weights)

    gc.collect()
    mem_gb = process.memory_info().rss / 1024**3

    print(f"{year}: Total income tax = ${total/1e9:.1f}B, Memory = {mem_gb:.2f} GB")

print("\nIf memory keeps growing significantly, we have a memory leak problem.")
