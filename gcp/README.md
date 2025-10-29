# GCP Jupyter Lab Setup Guide

Complete guide for creating and accessing a 64 GB spot instance with Python 3.13 and Jupyter Lab on Google Cloud Platform for PolicyEngine team members.

## Table of Contents
- [Creating the VM (Admin)](#creating-the-vm-admin)
- [Accessing Jupyter Lab (Windows)](#accessing-jupyter-lab-windows)
- [Accessing Jupyter Lab (Mac/Linux)](#accessing-jupyter-lab-maclinux)
- [Managing the VM](#managing-the-vm)
- [Cost Management](#cost-management)
- [Troubleshooting](#troubleshooting)

---

## Creating the VM (Admin)

### Prerequisites
- GCP Project: `policyengine-research`
- gcloud CLI installed and authenticated
- Appropriate IAM permissions (Compute Admin or equivalent)

### Option 1: Use the Script (Recommended)

```bash
./create_jupyter_spot.sh
```

The script takes ~10-15 minutes to complete (compiling Python 3.13 from source).

### Option 2: Manual Setup

See the `create_jupyter_spot.sh` script for the full command.

### VM Specifications (Spot Instance)

| Spec | Value |
|------|-------|
| Machine Type | n2-standard-16 |
| vCPUs | 16 |
| Memory | 64 GB |
| Boot Disk | 200 GB |
| Cost | ~$180/month (spot) vs $600/month (on-demand) |
| Python Version | 3.13 |
| Provisioning | Spot (may terminate after 24 hours) |

---

## Accessing Jupyter Lab (Windows)

### First Time Setup

1. **Install Google Cloud CLI**
   - Download from: https://cloud.google.com/sdk/docs/install
   - Run installer, follow prompts
   - Restart PowerShell/Command Prompt after installation

2. **Authenticate**
   ```powershell
   gcloud auth login
   gcloud config set project policyengine-research
   ```

3. **Verify access**
   ```powershell
   gcloud compute instances list --zone=us-central1-a
   ```
   You should see `jupyter-workstation` listed.

### Connect to Jupyter

1. **Open SSH tunnel** (in PowerShell or Command Prompt):
   ```powershell
   gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888
   ```

2. **Get the token** (in the SSH window that opened):
   ```bash
   jupyter server list
   ```
   Copy the token value from the output.

3. **Access Jupyter**
   - Open browser to: http://localhost:8888
   - Paste the token and log in

### Using Jupyter

The environment is pre-configured with:
- Python 3.13
- policyengine-us
- numpy, pandas, scipy, matplotlib
- Access to GCS bucket: `policyengine-calibration`

---

## Accessing Jupyter Lab (Mac/Linux)

### First Time Setup

1. **Install Google Cloud CLI** (if not already installed):
   ```bash
   # Mac with Homebrew
   brew install google-cloud-sdk

   # Or download from: https://cloud.google.com/sdk/docs/install
   ```

2. **Authenticate**
   ```bash
   gcloud auth login
   gcloud config set project policyengine-research
   ```

3. **Verify access**
   ```bash
   gcloud compute instances list --zone=us-central1-a
   ```

### Connect to Jupyter

1. **Open SSH tunnel**:
   ```bash
   gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888
   ```

2. **Get the token** (in the SSH window):
   ```bash
   jupyter server list
   ```

3. **Access Jupyter**:
   - Open browser to: http://localhost:8888
   - Paste the token and log in

---

## Managing the VM

### Start Jupyter (if not running)

SSH into the VM and run:
```bash
source /home/jupyter/pe-env/bin/activate
jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
```

### Installing Additional Packages

In a Jupyter notebook cell:
```python
# Fix permissions (only needed first time)
!sudo chown -R $(whoami):$(whoami) /home/jupyter/pe-env

# Install packages
!pip install package-name
```

### Access GCS Buckets

The VM has service account access. In a notebook:
```python
# List calibration data
!gsutil ls gs://policyengine-calibration/

# Download data
!gsutil cp gs://policyengine-calibration/path/to/file.pkl .
```

---

## Cost Management

### IMPORTANT: Stop VM When Not In Use

**To avoid unnecessary charges, always stop the VM when you're done working:**

```bash
gcloud compute instances stop jupyter-workstation --zone=us-central1-a
```

**Stopped VMs only incur storage costs (~$8/month for 200GB disk), not compute costs.**

### Start VM When Needed

```bash
gcloud compute instances start jupyter-workstation --zone=us-central1-a
```

After starting, wait ~30 seconds then connect via SSH tunnel.

### Check VM Status

```bash
gcloud compute instances list --zone=us-central1-a
```

Status values:
- `RUNNING` - VM is on and billing compute costs
- `TERMINATED` - VM is stopped, only billing storage
- `PROVISIONING` - VM is starting up

### Spot Instance Behavior

This is a **spot instance** which means:
- 70% cheaper than on-demand (~$180/month vs $600/month)
- GCP may terminate it after 24 hours or when capacity is needed
- When terminated, it automatically **stops** (not deleted)
- All data on disk is preserved
- Simply restart with the `start` command above

### Delete VM Permanently

**Only do this when completely done with the project:**

```bash
gcloud compute instances delete jupyter-workstation --zone=us-central1-a
```

⚠️ This deletes all data and cannot be undone.

### Cost Summary

| State | Compute Cost | Storage Cost | Total/Month |
|-------|--------------|--------------|-------------|
| Running (spot) | ~$172 | ~$8 | ~$180 |
| Running (on-demand) | ~$592 | ~$8 | ~$600 |
| Stopped | $0 | ~$8 | ~$8 |
| Deleted | $0 | $0 | $0 |

---

## Troubleshooting

### "Permission denied" when installing packages

Run in notebook cell:
```python
!sudo chown -R $(whoami):$(whoami) /home/jupyter/pe-env
```

### Can't connect to Jupyter / Connection refused

1. Check VM is running:
   ```bash
   gcloud compute instances list --zone=us-central1-a
   ```

2. If stopped, start it:
   ```bash
   gcloud compute instances start jupyter-workstation --zone=us-central1-a
   ```

3. SSH in and start Jupyter:
   ```bash
   gcloud compute ssh jupyter-workstation --zone=us-central1-a
   source /home/jupyter/pe-env/bin/activate
   jupyter lab --ip=0.0.0.0 --port=8888 --allow-root
   ```

### VM was terminated by GCP (spot instance)

This is expected behavior. Simply restart:
```bash
gcloud compute instances start jupyter-workstation --zone=us-central1-a
```

All your notebooks and data are preserved on disk.

### Need more memory

Create a new instance with `--machine-type=n1-highmem-16` (104 GB RAM) in the creation script.

### Token doesn't work

Generate a new token:
```bash
# SSH into VM
jupyter server list
# Or restart Jupyter to get fresh token
```

---

## Advanced: GPU Instance

For GPU-accelerated work (matches batch pipeline setup):

```bash
gcloud compute instances create jupyter-gpu \
    --zone=us-central1-a \
    --machine-type=n1-standard-16 \
    --accelerator=type=nvidia-tesla-p100,count=1 \
    --provisioning-model=SPOT \
    --instance-termination-action=STOP \
    --boot-disk-size=200GB \
    --image-family=pytorch-latest-gpu \
    --image-project=deeplearning-platform-release \
    --maintenance-policy=TERMINATE \
    --metadata="install-nvidia-driver=True" \
    --service-account=policyengine-research@policyengine-research.iam.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/cloud-platform
```

Access via SSH tunnel on port 8080:
```bash
gcloud compute ssh jupyter-gpu --zone=us-central1-a -- -L 8080:localhost:8080
```

---

## Related Documentation

This setup complements the batch processing pipeline at:
`policyengine_us_data/datasets/cps/geo_stacking_calibration/batch_pipeline/`

**Batch Pipeline**: Ephemeral VMs, containerized, GPU-optimized for automated L0 optimization jobs

**This Setup**: Persistent spot instance, interactive Jupyter, exploratory analysis and development

Both use the same GCP project (`policyengine-research`) and service account, with access to the same GCS buckets.
