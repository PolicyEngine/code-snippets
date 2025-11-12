# Jupyter Lab Quick Start

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

4. **Access**: Open http://localhost:8888 (no token needed)

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

4. **Access**: Open http://localhost:8888 (no token needed)

---

## Pre-installed Software

- Python 3.13 (via uv)
- jupyterlab, numpy, pandas, scipy, matplotlib
- uv (ultra-fast package manager)
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

### Need to install packages?

In notebook cell:
```python
!uv pip install package-name
```

### VM terminated by GCP?

Normal for spot instances. Just restart:
```bash
gcloud compute instances start jupyter-workstation --zone=us-central1-a
```

All your data is preserved.

### Can't connect?

Check VM is running:
```bash
gcloud compute instances list --zone=us-central1-a
```

---

Full documentation: See `README.md`
