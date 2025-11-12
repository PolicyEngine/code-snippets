# GitHub Self-Hosted Runners on GCP

Reusable scripts and documentation for setting up GitHub Actions self-hosted runners on GCP spot instances.

## Quick Start

### For Any Repository

```bash
cd ~/devl/code-snippets/gcp/github-runners
./create_runner.sh <org> <repo-name> <vm-name> [machine-type] [zone]
```

**Examples:**

```bash
# Default: 64GB RAM, us-central1-a
./create_runner.sh PolicyEngine policyengine-us-data github-runner

# Custom machine type: 32GB RAM
./create_runner.sh PolicyEngine policyengine-us-data pe-runner n2-standard-8

# Different zone
./create_runner.sh PolicyEngine another-repo runner-2 n2-standard-16 us-east1-b
```

## Machine Type Options

| Machine Type | RAM | vCPUs | Cost/Month Running | Cost/Month Stopped |
|--------------|-----|-------|-------------------|-------------------|
| n2-standard-4 | 16GB | 4 | ~$45 | ~$8 |
| n2-standard-8 | 32GB | 8 | ~$90 | ~$8 |
| n2-standard-16 | 64GB | 16 | ~$180 | ~$8 |

## Setup Process

### 1. Create the VM

```bash
./create_runner.sh PolicyEngine your-repo runner-name
```

Wait ~3 minutes for setup to complete.

### 2. Get GitHub Token

Go to your repository's runner settings:
```
https://github.com/YOUR_ORG/YOUR_REPO/settings/actions/runners/new
```

Select:
- **Runner image**: Linux
- **Architecture**: x64

Scroll to the **Configure** section and copy the token from the command.

### 3. SSH and Configure

```bash
# SSH into the VM
gcloud compute ssh <vm-name> --zone=<zone>

# Switch to runner user
sudo su - runner

# Configure with GitHub token
cd ~/actions-runner
./config.sh --url https://github.com/YOUR_ORG/YOUR_REPO --token YOUR_TOKEN

# Press Enter for all prompts (accept defaults)

# Exit back to your user
exit
```

### 4. Start the Service

```bash
# Install and start the runner service
sudo bash -c 'cd /home/runner/actions-runner && ./svc.sh install runner && ./svc.sh start && ./svc.sh status'
```

You should see: `active (running)`

### 5. Update Workflow

In your repository, update `.github/workflows/*.yaml`:

```yaml
jobs:
  test:
    runs-on: self-hosted  # Changed from: ubuntu-latest
```

### 6. Verify

Check that runner appears online:
```
https://github.com/YOUR_ORG/YOUR_REPO/settings/actions/runners
```

You should see a green dot next to your runner name.

## Cost Savings: Multiple Runners on One VM

Instead of creating separate VMs for each repo, you can run **multiple runner services on a single large VM** to save costs.

**Example: Our Current Setup**
- One VM (`github-runner`, 64GB RAM)
- Two runners:
  - `/home/runner/actions-runner` → policyengine-us-data
  - `/home/runner/actions-runner-us` → policyengine-us (named `gcp-runner-us`)
- **Savings**: ~$180/month instead of ~$360/month

### Adding a Second Runner to Existing VM

**Option 1: Use the helper script (easiest)**

```bash
cd ~/devl/code-snippets/gcp/github-runners
./add_runner_to_vm.sh github-runner PolicyEngine policyengine-us us gcp-runner-us

# It will prompt you for the GitHub token
```

**Option 2: Manual setup**

```bash
# SSH into existing VM
gcloud compute ssh github-runner --zone=us-central1-a

# Download and set up second runner (as root)
sudo su
mkdir -p /home/runner/actions-runner-REPONAME
cd /home/runner/actions-runner-REPONAME

RUNNER_VERSION=$(curl -s https://api.github.com/repos/actions/runner/releases/latest | grep -oP 'tag_name": "v\K(.*)(?=")')
curl -o actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz -L https://github.com/actions/runner/releases/download/v${RUNNER_VERSION}/actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
tar xzf actions-runner-linux-x64-${RUNNER_VERSION}.tar.gz
chown -R runner:runner /home/runner/actions-runner-REPONAME
exit

# Configure as runner user
sudo su - runner
cd ~/actions-runner-REPONAME
./config.sh --url https://github.com/ORG/REPO --token YOUR_TOKEN --name gcp-runner-REPONAME
exit

# Start the service
sudo /home/runner/actions-runner-REPONAME/svc.sh install runner
sudo /home/runner/actions-runner-REPONAME/svc.sh start
sudo /home/runner/actions-runner-REPONAME/svc.sh status
```

**Important**: Give each runner directory a unique name (e.g., `actions-runner-us`, `actions-runner-data`) and use descriptive runner names (`--name gcp-runner-us`) for easy identification in GitHub UI.

## Daily Usage

See [MANAGEMENT.md](./MANAGEMENT.md) for:
- Starting/stopping VMs
- Scaling machine types
- Cost monitoring
- Troubleshooting
- Re-registering runners

## Common Issues

### Runner Not Picking Up Jobs

1. Check runner status online: `https://github.com/ORG/REPO/settings/actions/runners`
2. SSH and check service: `sudo /home/runner/actions-runner/svc.sh status`
3. Restart if needed: `sudo /home/runner/actions-runner/svc.sh restart`

### Token Expired

Tokens expire after ~1 hour. Get a new one from the GitHub runner setup page and reconfigure:

```bash
sudo su - runner
cd ~/actions-runner
./config.sh --url https://github.com/ORG/REPO --token NEW_TOKEN
exit
sudo bash -c 'cd /home/runner/actions-runner && ./svc.sh install runner && ./svc.sh start'
```

### Permission Denied

Always use `sudo` or `sudo su` to access `/home/runner/actions-runner/`.

## Files

- `create_runner.sh` - Main script to create runner VMs (parameterized for any repo)
- `add_runner_to_vm.sh` - Helper script to add additional runners to existing VM
- `MANAGEMENT.md` - Day-to-day operations guide
- `examples/` - Example configurations for specific repos

## Infrastructure Details

- **GCP Project**: policyengine-research
- **Service Account**: policyengine-research@policyengine-research.iam.gserviceaccount.com
- **Instance Type**: Spot instances (70% cost savings)
- **Disk**: 200GB standard persistent disk
- **Swap**: 32GB swap file for memory-intensive builds
- **Software**: Docker, uv, GitHub Actions runner

## Security Notes

- Runners use GCP service account authentication (Workload Identity)
- Spot instances auto-stop (not terminate) when preempted
- Self-hosted runners in public repos can run potentially dangerous code from PRs
- Consider using environment protection rules or requiring approval for PR runs
