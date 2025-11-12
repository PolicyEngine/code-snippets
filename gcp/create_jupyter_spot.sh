#!/bin/bash

set -e

echo "Creating Jupyter Lab spot instance with 64 GB RAM and uv..."

gcloud compute instances create jupyter-workstation \
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
    --tags=jupyter-server \
    --metadata=startup-script='#!/bin/bash
apt-get update
apt-get install -y curl git

# Create 32GB swap file for memory-intensive workloads
fallocate -l 32G /swapfile
chmod 600 /swapfile
mkswap /swapfile
swapon /swapfile
echo '/swapfile none swap sw 0 0' >> /etc/fstab
sysctl vm.swappiness=10
echo 'vm.swappiness=10' >> /etc/sysctl.conf

# Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh
source /root/.cargo/env

# Create jupyter user and directory
useradd -m -s /bin/bash jupyter || true
mkdir -p /home/jupyter
chown -R jupyter:jupyter /home/jupyter

# Set up Python 3.13 environment with uv
cd /home/jupyter
export PATH="/root/.cargo/bin:$PATH"
uv venv --python 3.13 .venv
source .venv/bin/activate

# Install packages with uv (much faster than pip!)
uv pip install jupyterlab numpy pandas scipy matplotlib

# Generate Jupyter config
mkdir -p /home/jupyter/.jupyter
cat > /home/jupyter/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = False
c.ServerApp.root_dir = "/home/jupyter"
c.ServerApp.password = ""
c.ServerApp.token = ""
c.ServerApp.password_required = False
c.ServerApp.allow_origin = "*"
EOF

# Set up shell initialization for auto-activation
cat > /home/jupyter/.bashrc << EOF
# Auto-activate uv Python environment
if [ -f /home/jupyter/.venv/bin/activate ]; then
    source /home/jupyter/.venv/bin/activate
fi

# Add uv to PATH
export PATH="/root/.cargo/bin:\$PATH"

# Helpful aliases
alias ll="ls -la"
alias jp="jupyter lab"
alias uv="/root/.cargo/bin/uv"
EOF

# Create systemd service for Jupyter
cat > /etc/systemd/system/jupyter.service << EOF
[Unit]
Description=Jupyter Lab Server
After=network.target

[Service]
Type=simple
User=jupyter
WorkingDirectory=/home/jupyter
Environment="PATH=/home/jupyter/.venv/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"
ExecStart=/home/jupyter/.venv/bin/jupyter lab --config=/home/jupyter/.jupyter/jupyter_lab_config.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
EOF

# Fix permissions
chown -R jupyter:jupyter /home/jupyter
chmod 755 /home/jupyter/.venv/bin/*

# Enable and start Jupyter service
systemctl daemon-reload
systemctl enable jupyter.service
systemctl start jupyter.service

# Wait a moment for the service to start
sleep 5

# Get the initial token
su - jupyter -c "source /home/jupyter/.venv/bin/activate && jupyter server list" > /home/jupyter/initial_token.txt 2>&1 || true
'

echo ""
echo "✓ Instance created successfully!"
echo ""
echo "Setup will complete in ~2 minutes (much faster with uv!)"
echo ""
echo "Next steps:"
echo "1. Wait ~2 minutes for setup to complete"
echo "2. Connect via SSH tunnel: gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888"
echo "3. Open browser to: http://localhost:8888"
echo "4. The Python environment with Python 3.13 will auto-activate on SSH"
echo ""
echo "⚠️  IMPORTANT - Cost Management:"
echo "   Stop VM when done: gcloud compute instances stop jupyter-workstation --zone=us-central1-a"
echo "   Stopped VMs only cost ~\$8/month for storage (vs ~\$180/month running)"