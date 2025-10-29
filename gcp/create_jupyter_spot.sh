#!/bin/bash

set -e

echo "Creating Jupyter Lab spot instance with 64 GB RAM..."

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
apt-get install -y wget build-essential libssl-dev zlib1g-dev \
    libncurses5-dev libncursesw5-dev libreadline-dev libsqlite3-dev \
    libgdbm-dev libdb5.3-dev libbz2-dev libexpat1-dev liblzma-dev \
    tk-dev libffi-dev git

cd /tmp
wget https://www.python.org/ftp/python/3.13.0/Python-3.13.0.tgz
tar xzf Python-3.13.0.tgz
cd Python-3.13.0
./configure --enable-optimizations
make -j $(nproc)
make altinstall

mkdir -p /home/jupyter
cd /home/jupyter
python3.13 -m venv pe-env
source pe-env/bin/activate
pip install --upgrade pip
pip install jupyterlab numpy pandas scipy matplotlib

jupyter lab --generate-config
cat >> /root/.jupyter/jupyter_lab_config.py << EOF
c.ServerApp.ip = "0.0.0.0"
c.ServerApp.port = 8888
c.ServerApp.open_browser = False
c.ServerApp.allow_root = True
EOF
'

echo ""
echo "✓ Instance created successfully!"
echo ""
echo "Startup script is running (will take ~10-15 minutes to compile Python 3.13)"
echo ""
echo "Next steps:"
echo "1. Wait for startup script to complete (~10-15 minutes)"
echo "2. Connect via SSH tunnel: gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888"
echo "3. Get token: jupyter server list"
echo "4. Open browser to: http://localhost:8888"
echo ""
echo "⚠️  IMPORTANT - Cost Management:"
echo "   Stop VM when done: gcloud compute instances stop jupyter-workstation --zone=us-central1-a"
echo "   Stopped VMs only cost ~\$8/month for storage (vs ~\$180/month running)"
