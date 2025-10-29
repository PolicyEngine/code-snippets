# Quick Start Guide - Jupyter Lab on GCP

## For New Users

### Windows Users

1. **Install gcloud** (one-time): https://cloud.google.com/sdk/docs/install

2. **Authenticate** (one-time):
   ```powershell
   gcloud auth login
   gcloud config set project policyengine-research
   ```

3. **Connect to Jupyter**:
   ```powershell
   gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888
   ```

4. **Get token** (in SSH window):
   ```bash
   jupyter server list
   ```

5. **Access**: Open http://localhost:8888 and paste token

### Mac/Linux Users

1. **Install gcloud** (one-time):
   ```bash
   brew install google-cloud-sdk
   ```

2. **Authenticate** (one-time):
   ```bash
   gcloud auth login
   gcloud config set project policyengine-research
   ```

3. **Connect to Jupyter**:
   ```bash
   gcloud compute ssh jupyter-workstation --zone=us-central1-a -- -L 8888:localhost:8888
   ```

4. **Get token** (in SSH window):
   ```bash
   jupyter server list
   ```

5. **Access**: Open http://localhost:8888 and paste token

---

## Pre-installed Software

- Python 3.13
- policyengine-us
- numpy, pandas, scipy, matplotlib
- Access to GCS bucket: `policyengine-calibration`

---

## Important Commands

### If VM is stopped, start it:
```bash
gcloud compute instances start jupyter-workstation --zone=us-central1-a
```

### **ALWAYS stop VM when done to save money:**
```bash
gcloud compute instances stop jupyter-workstation --zone=us-central1-a
```

### Check VM status:
```bash
gcloud compute instances list --zone=us-central1-a
```

---

## Cost Warning

- **Running**: ~$180/month (spot instance)
- **Stopped**: ~$8/month (storage only)

**Always stop the VM when you're done working!**

---

## Troubleshooting

### Can't install packages (permission denied)?

In notebook cell:
```python
!sudo chown -R $(whoami):$(whoami) /home/jupyter/pe-env
!pip install package-name
```

### VM terminated by GCP?

Normal for spot instances. Just restart:
```bash
gcloud compute instances start jupyter-workstation --zone=us-central1-a
```

All your data is preserved.

### Token doesn't work?

Get fresh token:
```bash
jupyter server list
```

---

Full documentation: See `README.md`
