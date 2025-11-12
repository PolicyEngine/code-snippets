# GitHub Runner Management

## Quick Reference

### Start/Stop Runner
```bash
# Stop when not developing (saves ~$180/month)
gcloud compute instances stop github-runner --zone=us-central1-a

# Start when needed
gcloud compute instances start github-runner --zone=us-central1-a

# Check status
gcloud compute instances describe github-runner --zone=us-central1-a --format="value(status)"
```

### SSH Access
```bash
gcloud compute ssh github-runner --zone=us-central1-a
```

### Check Runner Services

**Current setup: Two runners on one VM**
- policyengine-us-data: `/home/runner/actions-runner`
- policyengine-us: `/home/runner/actions-runner-us`

```bash
# SSH into VM first, then:

# Check policyengine-us-data runner
sudo systemctl status actions.runner.PolicyEngine-policyengine-us-data.github-runner.service

# Check policyengine-us runner
sudo systemctl status actions.runner.PolicyEngine-policyengine-us.gcp-runner-us.service

# Or check both at once
sudo systemctl status 'actions.runner.*'
```

**Quick service commands:**
```bash
# Restart specific runner
sudo systemctl restart actions.runner.PolicyEngine-policyengine-us-data.github-runner.service

# Stop specific runner
sudo systemctl stop actions.runner.PolicyEngine-policyengine-us.gcp-runner-us.service

# View logs
sudo journalctl -u actions.runner.PolicyEngine-policyengine-us-data.github-runner.service -f
```

## Scaling Machine Size

If runner runs out of memory, scale up the machine type:

```bash
# Stop the VM
gcloud compute instances stop github-runner --zone=us-central1-a

# Change machine type (examples)
gcloud compute instances set-machine-type github-runner \
    --zone=us-central1-a \
    --machine-type=n2-standard-8   # 32GB RAM

# Or for more memory:
gcloud compute instances set-machine-type github-runner \
    --zone=us-central1-a \
    --machine-type=n2-standard-16  # 64GB RAM

# Restart
gcloud compute instances start github-runner --zone=us-central1-a
```

## Troubleshooting

### Runner Not Picking Up Jobs

**For policyengine-us-data runner:**
1. Check runner online: https://github.com/PolicyEngine/policyengine-us-data/settings/actions/runners
2. Check service: `sudo systemctl status actions.runner.PolicyEngine-policyengine-us-data.github-runner.service`
3. Restart: `sudo systemctl restart actions.runner.PolicyEngine-policyengine-us-data.github-runner.service`

**For policyengine-us runner:**
1. Check runner online: https://github.com/PolicyEngine/policyengine-us/settings/actions/runners (requires admin)
2. Check service: `sudo systemctl status actions.runner.PolicyEngine-policyengine-us.gcp-runner-us.service`
3. Restart: `sudo systemctl restart actions.runner.PolicyEngine-policyengine-us.gcp-runner-us.service`

### Re-register Runner
If token expires or runner needs reconfiguration:

**Replace `RUNNER_DIR` with either:**
- `actions-runner` for policyengine-us-data
- `actions-runner-us` for policyengine-us

```bash
# SSH into VM
gcloud compute ssh github-runner --zone=us-central1-a

# Stop service
sudo /home/runner/RUNNER_DIR/svc.sh stop
sudo /home/runner/RUNNER_DIR/svc.sh uninstall

# Get new token from GitHub:
# policyengine-us-data: https://github.com/PolicyEngine/policyengine-us-data/settings/actions/runners/new
# policyengine-us: https://github.com/PolicyEngine/policyengine-us/settings/actions/runners/new (requires admin)

# Reconfigure as runner user
sudo su - runner
cd ~/RUNNER_DIR
./config.sh --url https://github.com/PolicyEngine/REPO_NAME --token YOUR_NEW_TOKEN --name RUNNER_NAME

# Reinstall service
exit  # back to your user
sudo /home/runner/RUNNER_DIR/svc.sh install runner
sudo /home/runner/RUNNER_DIR/svc.sh start
sudo /home/runner/RUNNER_DIR/svc.sh status
```

**Example for policyengine-us:**
```bash
sudo /home/runner/actions-runner-us/svc.sh stop
sudo /home/runner/actions-runner-us/svc.sh uninstall
sudo su - runner
cd ~/actions-runner-us
./config.sh --url https://github.com/PolicyEngine/policyengine-us --token YOUR_TOKEN --name gcp-runner-us
exit
sudo /home/runner/actions-runner-us/svc.sh install runner
sudo /home/runner/actions-runner-us/svc.sh start
```

## Monitoring Costs

### Current VM Costs (approximate)
- **n2-standard-4** (16GB): ~$45/month running, ~$8/month stopped
- **n2-standard-8** (32GB): ~$90/month running, ~$8/month stopped
- **n2-standard-16** (64GB): ~$180/month running, ~$8/month stopped

### Set Up Billing Alert
```bash
# Check current month costs
gcloud billing projects describe policyengine-research --format="value(billingAccountName)"
```

## VM Deletion

If you want to completely remove the runner:

```bash
# Remove runner from GitHub first (via web UI)
# https://github.com/PolicyEngine/policyengine-us-data/settings/actions/runners

# Delete VM
gcloud compute instances delete github-runner --zone=us-central1-a
```
