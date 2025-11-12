#!/bin/bash

set -e

# Usage: ./add_runner_to_vm.sh <vm-name> <repo-org> <repo-name> <runner-dir-suffix> <runner-name> [zone]
# Example: ./add_runner_to_vm.sh github-runner PolicyEngine policyengine-us us gcp-runner-us

if [ "$#" -lt 5 ]; then
    echo "Usage: $0 <vm-name> <repo-org> <repo-name> <runner-dir-suffix> <runner-name> [zone]"
    echo ""
    echo "This script helps you add a second (or third, etc.) runner to an existing VM."
    echo ""
    echo "Parameters:"
    echo "  vm-name            - Name of existing VM (e.g., github-runner)"
    echo "  repo-org           - GitHub org (e.g., PolicyEngine)"
    echo "  repo-name          - GitHub repo (e.g., policyengine-us)"
    echo "  runner-dir-suffix  - Directory suffix (e.g., 'us' creates /home/runner/actions-runner-us)"
    echo "  runner-name        - Name to display in GitHub (e.g., gcp-runner-us)"
    echo "  zone               - GCP zone (default: us-central1-a)"
    echo ""
    echo "Example:"
    echo "  $0 github-runner PolicyEngine policyengine-us us gcp-runner-us"
    echo ""
    echo "This will create /home/runner/actions-runner-us for the policyengine-us repository."
    echo ""
    exit 1
fi

VM_NAME=$1
REPO_ORG=$2
REPO_NAME=$3
DIR_SUFFIX=$4
RUNNER_NAME=$5
ZONE=${6:-us-central1-a}

REPO_URL="https://github.com/${REPO_ORG}/${REPO_NAME}"
RUNNER_DIR="/home/runner/actions-runner-${DIR_SUFFIX}"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "Adding runner to existing VM"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  VM: ${VM_NAME} (${ZONE})"
echo "  Repository: ${REPO_URL}"
echo "  Runner Directory: ${RUNNER_DIR}"
echo "  Runner Name: ${RUNNER_NAME}"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

echo "Step 1: Get GitHub runner token"
echo "       Go to: ${REPO_URL}/settings/actions/runners/new"
echo "       Copy the token from the config command"
echo ""
read -p "Enter the GitHub runner token: " GITHUB_TOKEN

if [ -z "$GITHUB_TOKEN" ]; then
    echo "Error: Token cannot be empty"
    exit 1
fi

echo ""
echo "Step 2: Creating runner directory and downloading software..."
echo ""

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
set -e
sudo mkdir -p ${RUNNER_DIR}
cd ${RUNNER_DIR}
RUNNER_VERSION=\$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -oP 'tag_name\": \"v\K(.*)(?=\")')
echo \"Downloading runner version \${RUNNER_VERSION}...\"
sudo curl -sL -o actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz https://github.com/actions/runner/releases/download/v\${RUNNER_VERSION}/actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz
sudo tar xzf actions-runner-linux-x64-\${RUNNER_VERSION}.tar.gz
sudo chown -R runner:runner ${RUNNER_DIR}
echo \"✓ Runner software downloaded\"
"

echo ""
echo "Step 3: Configuring runner..."
echo ""

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
set -e
sudo su - runner -c 'cd ${RUNNER_DIR} && ./config.sh --url ${REPO_URL} --token ${GITHUB_TOKEN} --name ${RUNNER_NAME}'
"

echo ""
echo "Step 4: Installing and starting service..."
echo ""

gcloud compute ssh ${VM_NAME} --zone=${ZONE} --command="
set -e
sudo ${RUNNER_DIR}/svc.sh install runner
sudo ${RUNNER_DIR}/svc.sh start
sudo ${RUNNER_DIR}/svc.sh status
"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✓ Runner added successfully!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Runner: ${RUNNER_NAME}"
echo "Directory: ${RUNNER_DIR}"
echo "Repository: ${REPO_URL}"
echo ""
echo "Verify runner is online at:"
echo "${REPO_URL}/settings/actions/runners"
echo ""
echo "Service name:"
echo "  sudo systemctl status actions.runner.${REPO_ORG}-${REPO_NAME}.${RUNNER_NAME}.service"
echo ""
