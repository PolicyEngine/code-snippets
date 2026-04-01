# Dev Workstation Quick Start

## Create the VM (one-time, admin)

```bash
./create_dev_workstation.sh                    # defaults: dev-workstation, 200GB disk
./create_dev_workstation.sh my-vm 128          # custom name, 128GB disk
```

Wait ~2 minutes for startup script to finish.

---

## Granting Access to Team Members (admin)

### IAM roles needed

Grant these roles to the team member's Google email in [IAM & Admin > IAM](https://console.cloud.google.com/iam-admin/iam?project=policyengine-research):

- **Compute OS Login** (`roles/compute.osLogin`) — allows SSH access
- **Service Account User** (`roles/iam.serviceAccountUser`) — allows using the research service account

Or alternatively: **Compute Instance Admin (v1)** (`roles/compute.instanceAdmin.v1`) for full instance management.

### What the team member needs locally

1. **gcloud CLI** — https://cloud.google.com/sdk/docs/install
2. **VS Code** with the **Remote - SSH** extension

### Their first-time auth flow

```bash
gcloud auth login
gcloud config set project policyengine-research
gcloud compute config-ssh
```

No GCP Console access needed. Works fine on low-spec machines (8GB Mac, etc.) since all compute happens on the VM.

---

## Connect with VS Code

### First-time setup

1. Install VS Code extension: **Remote - SSH**
2. Run `gcloud compute config-ssh` (generates `~/.ssh/config` entries)
3. In VS Code: `Ctrl+Shift+P` > **Remote-SSH: Connect to Host** > `dev-workstation.us-central1-a.policyengine-research`

### Daily connect

1. Start the VM if stopped:
   ```bash
   gcloud compute instances start dev-workstation --zone=us-central1-a
   ```
2. In VS Code: `Ctrl+Shift+P` > **Remote-SSH: Connect to Host** > `dev-workstation.us-central1-a.policyengine-research`

**Note:** If the VM's external IP changed (happens on stop/start), re-run `gcloud compute config-ssh` before connecting.

---

## Connect via Terminal

```bash
gcloud compute ssh dev-workstation --zone=us-central1-a
```

---

## Pre-installed Software

| Software | Notes |
|----------|-------|
| git | Version control |
| gh | GitHub CLI — `gh auth login` on first use |
| uv | Fast Python package manager, system-wide |
| tmux | Terminal multiplexer (mouse on, 50k history) |
| build-essential | C/C++ compiler toolchain for native extensions |
| curl, unzip | Utilities |

---

## Common Workflows

### Clone a repo and set up a venv

```bash
gh auth login
git clone https://github.com/PolicyEngine/policyengine-us.git
cd policyengine-us
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### Run a long process in tmux

```bash
tmux new -s build
# run your command, then detach with Ctrl+B, D
# reattach later:
tmux attach -t build
```

---

## Stop / Start / Delete

```bash
# ALWAYS stop when done to save money
gcloud compute instances stop dev-workstation --zone=us-central1-a

# Start when needed
gcloud compute instances start dev-workstation --zone=us-central1-a

# Check status
gcloud compute instances list --zone=us-central1-a

# Delete permanently (destroys all data)
gcloud compute instances delete dev-workstation --zone=us-central1-a
```

---

## Cost

| State | Cost |
|-------|------|
| Running | ~$0.39/hr (~$280/month 24/7) |
| Stopped | ~$3-5/month (disk only) |
| Deleted | $0 |

This is an **on-demand** instance (not spot) — it will not be preempted.

---

## Troubleshooting

### VS Code can't connect

1. Check VM is running: `gcloud compute instances list --zone=us-central1-a`
2. Re-run `gcloud compute config-ssh` (IP may have changed after stop/start)
3. Try terminal SSH first to verify connectivity: `gcloud compute ssh dev-workstation --zone=us-central1-a`

### Startup script didn't finish

Check for the completion marker:
```bash
ls /opt/dev/.setup_complete
```

If missing, check startup script logs:
```bash
sudo journalctl -u google-startup-scripts.service
```

### Need more CPU/RAM

Stop the VM, resize, then restart:
```bash
gcloud compute instances stop dev-workstation --zone=us-central1-a
gcloud compute instances set-machine-type dev-workstation --zone=us-central1-a --machine-type=n2-standard-16
gcloud compute instances start dev-workstation --zone=us-central1-a
```

---

Full documentation: See `README.md`
