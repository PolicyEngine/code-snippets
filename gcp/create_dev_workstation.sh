#!/bin/bash

set -e

VM_NAME="${1:-dev-$(whoami)}"
DISK_SIZE="${2:-200}"

if gcloud compute instances describe "${VM_NAME}" --zone=us-central1-a &>/dev/null; then
    echo "Instance '${VM_NAME}' already exists. Deleting..."
    gcloud compute instances delete "${VM_NAME}" --zone=us-central1-a --quiet
fi

echo "Creating on-demand dev workstation '${VM_NAME}' with ${DISK_SIZE}GB disk..."

gcloud compute instances create "${VM_NAME}" \
    --zone=us-central1-a \
    --machine-type=n2-standard-8 \
    --provisioning-model=STANDARD \
    --boot-disk-size="${DISK_SIZE}GB" \
    --boot-disk-type=pd-standard \
    --image-family=ubuntu-2404-lts-amd64 \
    --image-project=ubuntu-os-cloud \
    --service-account=policyengine-research@policyengine-research.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform \
    --tags=dev-workstation \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y curl git tmux build-essential unzip

# Install GitHub CLI
curl -fsSL https://cli.github.com/packages/githubcli-archive-keyring.gpg | dd of=/usr/share/keyrings/githubcli-archive-keyring.gpg
chmod go+r /usr/share/keyrings/githubcli-archive-keyring.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/githubcli-archive-keyring.gpg] https://cli.github.com/packages stable main" | tee /etc/apt/sources.list.d/github-cli-stable.list > /dev/null
apt-get update
apt-get install -y gh

# Create 16GB swap file (half of 32GB RAM)
fallocate -l 16G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo "/swapfile none swap sw 0 0" >> /etc/fstab
sysctl vm.swappiness=10
echo "vm.swappiness=10" >> /etc/sysctl.conf

# Install uv system-wide
curl -LsSf https://astral.sh/uv/install.sh | CARGO_HOME=/usr/local sh

# Ensure uv is on PATH for all users
cat > /etc/profile.d/dev-env.sh << "EOF"
export PATH="/usr/local/bin:$PATH"
EOF

# Basic tmux config
cat > /etc/tmux.conf << "EOF"
set -g mouse on
set -g history-limit 50000
set -g default-terminal "screen-256color"
EOF

# Mark setup complete
mkdir -p /opt/dev
touch /opt/dev/.setup_complete
'

echo ""
echo "Instance '${VM_NAME}' created successfully!"
echo ""
echo "Setup will complete in ~2 minutes."
echo ""
echo "=== VS Code Remote SSH (recommended) ==="
echo "1. Run: gcloud compute config-ssh"
echo "2. Install VS Code extension: Remote - SSH"
echo "3. In VS Code: Ctrl+Shift+P > 'Remote-SSH: Connect to Host' > ${VM_NAME}.us-central1-a.policyengine-research"
echo ""
echo "=== Terminal SSH ==="
echo "gcloud compute ssh ${VM_NAME} --zone=us-central1-a"
echo ""
echo "=== VM Management ==="
echo "Stop:   gcloud compute instances stop ${VM_NAME} --zone=us-central1-a"
echo "Start:  gcloud compute instances start ${VM_NAME} --zone=us-central1-a"
echo "Delete: gcloud compute instances delete ${VM_NAME} --zone=us-central1-a"
echo ""
echo "=== Cost ==="
echo "Running: ~\$0.39/hr (~\$280/month if left on 24/7)"
echo "Stopped: ~\$3-5/month (disk storage only)"
echo "ALWAYS stop the VM when done!"
