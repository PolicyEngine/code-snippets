#!/bin/bash

set -e

# Usage: ./create_runner.sh <repo-org> <repo-name> <vm-name> [machine-type] [zone]
# Example: ./create_runner.sh PolicyEngine policyengine-us-data github-runner n2-standard-16 us-central1-a

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <repo-org> <repo-name> <vm-name> [machine-type] [zone]"
    echo ""
    echo "Examples:"
    echo "  $0 PolicyEngine policyengine-us-data github-runner"
    echo "  $0 PolicyEngine policyengine-us-data pe-runner n2-standard-8 us-central1-a"
    echo ""
    echo "Default machine-type: n2-standard-16 (64GB RAM)"
    echo "Default zone: us-central1-a"
    exit 1
fi

REPO_ORG=$1
REPO_NAME=$2
VM_NAME=$3
MACHINE_TYPE=${4:-n2-standard-16}
ZONE=${5:-us-central1-a}

REPO_URL="https://github.com/${REPO_ORG}/${REPO_NAME}"

echo "Creating GitHub Actions self-hosted runner..."
echo "  Repository: ${REPO_URL}"
echo "  VM Name: ${VM_NAME}"
echo "  Machine Type: ${MACHINE_TYPE}"
echo "  Zone: ${ZONE}"
echo ""

gcloud compute instances create ${VM_NAME} \
    --zone=${ZONE} \
    --machine-type=${MACHINE_TYPE} \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=200GB \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --service-account=policyengine-research@policyengine-research.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=github-runner \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y curl git build-essential

# Create 32GB swap file for memory-intensive workloads
fallocate -l 32G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '\''/swapfile none swap sw 0 0'\'' >> /etc/fstab
sysctl vm.swappiness=10
echo '\''vm.swappiness=10'\'' >> /etc/sysctl.conf

# Install Docker (required for many GitHub Actions)
curl -fsSL https://get.docker.com -o get-docker.sh
sh get-docker.sh
usermod -aG docker ubuntu

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create runner user and directory
useradd -m -s /bin/bash runner || true
mkdir -p /home/runner/actions-runner
chown -R runner:runner /home/runner

# Install GitHub Actions runner
cd /home/runner/actions-runner
RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -oP '\''tag_name": "v\K(.*)(?=")'\'')
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
tar xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
chown -R runner:runner /home/runner/actions-runner

# Add runner to docker group
usermod -aG docker runner

# Set up marker file for post-setup
touch /home/runner/runner_installed
chown runner:runner /home/runner/runner_installed

echo "Runner software installed. Manual configuration required - see instance logs for instructions."
'

echo ""
echo "✓ Instance created successfully!"
echo ""
echo "Setup will complete in ~3 minutes"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "NEXT STEPS - Configure the runner:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "1. Wait ~3 minutes for VM setup to complete"
echo ""
echo "2. Get a runner registration token from GitHub:"
echo "   ${REPO_URL}/settings/actions/runners/new"
echo ""
echo "3. SSH into the instance:"
echo "   gcloud compute ssh ${VM_NAME} --zone=${ZONE}"
echo ""
echo "4. Configure the runner (in SSH session):"
echo "   sudo su - runner"
echo "   cd ~/actions-runner"
echo "   ./config.sh --url ${REPO_URL} --token YOUR_TOKEN_HERE"
echo "   # Press Enter for all prompts to accept defaults"
echo "   exit"
echo ""
echo "5. Install and start the service (back as your user):"
echo "   sudo bash -c 'cd /home/runner/actions-runner && ./svc.sh install runner && ./svc.sh start && ./svc.sh status'"
echo ""
echo "6. Verify runner appears online at:"
echo "   ${REPO_URL}/settings/actions/runners"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "COST MANAGEMENT:"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "Stop VM:  gcloud compute instances stop ${VM_NAME} --zone=${ZONE}"
echo "Start VM: gcloud compute instances start ${VM_NAME} --zone=${ZONE}"
echo ""
echo "Machine type: ${MACHINE_TYPE}"
case ${MACHINE_TYPE} in
    n2-standard-4)
        echo "Cost: ~\$45/month running, ~\$8/month stopped"
        ;;
    n2-standard-8)
        echo "Cost: ~\$90/month running, ~\$8/month stopped"
        ;;
    n2-standard-16)
        echo "Cost: ~\$180/month running, ~\$8/month stopped"
        ;;
    *)
        echo "Cost: Check GCP pricing for ${MACHINE_TYPE}"
        ;;
esac
echo ""
