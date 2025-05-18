#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

echo "--- Starting the script ---"

# --- Configuration ---
# Replace these with the actual repository URLs
CLONE_DIR="/app/cloned_repos" # A directory inside the container to clone into

# --- Create clone directory ---
mkdir -p "${CLONE_DIR}"
cd "${CLONE_DIR}"
echo "--- Cloning repositories into ${CLONE_DIR} ---"

# --- Clone Repositories ---
git clone --progress https://github.com/PolicyEngine/policyengine-us-data.git
git clone --progress https://github.com/PolicyEngine/policyengine-core.git
git clone --progress https://github.com/PolicyEngine/policyengine-us.git

echo "--- Repositories cloned successfully ---"

echo "--- Install repo packages jointly ---"
uv venv pe
source pe/bin/activate
uv pip install -e ./policyengine-us-data \
                -e ./policyengine-core \
                -e ./policyengine-us

# A branch is out to fix the setuptools dependency in -us-data, but not sure about tables
uv pip install setuptools
uv pip install tables
uv pip install torch

# --- Run your code ---
cd policyengine-us-data 

# Replace the github download utility from another branch, so it doesn't need the token
git fetch origin
git checkout origin/fix/210-add-python312-ci -- policyengine_us_data/utils/github.py

# us-data downloads and processing
make download
# TODO: figure out how to pass in hugging face token as environment variable in docker run
make data

echo "--- Run the testing portion ---"
python check_estimates.py
echo "--- Script finished successfully. But check the tests with your own eyes! ---"
