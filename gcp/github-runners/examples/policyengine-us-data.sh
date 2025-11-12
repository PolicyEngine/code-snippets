#!/bin/bash

set -e

echo "Creating GitHub Actions self-hosted runner with 64 GB RAM..."

gcloud compute instances create github-runner \
    --zone=us-central1-a \
    --machine-type=n2-standard-16 \
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
echo "REQUIRED: Configure the runner with these commands:"
echo ""
echo "1. SSH into the instance:"
echo "   gcloud compute ssh github-runner --zone=us-central1-a"
echo ""
echo "2. Get a runner registration token from GitHub:"
echo "   Go to: https://github.com/PolicyEngine/policyengine-us-data/settings/actions/runners/new"
echo "   Copy the token from the configuration command"
echo ""
echo "3. Switch to runner user and configure:"
echo "   sudo su - runner"
echo "   cd ~/actions-runner"
echo "   ./config.sh --url https://github.com/PolicyEngine/policyengine-us-data --token YOUR_TOKEN_HERE"
echo ""
echo "4. Install and start the runner service (as root):"
echo "   exit  # back to ubuntu user"
echo "   cd /home/runner/actions-runner"
echo "   sudo ./svc.sh install runner"
echo "   sudo ./svc.sh start"
echo ""
echo "5. Verify runner is running:"
echo "   sudo ./svc.sh status"
echo ""
echo "⚠️  IMPORTANT - Cost Management:"
echo "   Stop VM when not developing: gcloud compute instances stop github-runner --zone=us-central1-a"
echo "   Start VM when needed:       gcloud compute instances start github-runner --zone=us-central1-a"
echo "   Stopped VMs only cost ~\$8/month for storage (vs ~\$180/month running)"
