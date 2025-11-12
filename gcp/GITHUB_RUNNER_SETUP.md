# GitHub Self-Hosted Runner Setup on GCP

## Problem
GitHub Actions failing with "operation was canceled" after ~17 minutes during `make data` step. Standard `ubuntu-latest` runners hitting memory/resource limits during dataset build.

## Current Infrastructure
- GCP project: `policyengine-research`
- Service account: `policyengine-research@policyengine-research.iam.gserviceaccount.com`
- Workload Identity Federation already configured
- Experience with n2-standard-16 (64GB RAM) spot instances
- Existing `create_jupyter_spot.sh` script as template

## Difficulty Assessment: MODERATE (2-3 hours setup)

### Advantages
1. GCP service accounts already in use - auth solved
2. `create_jupyter_spot.sh` is 90% of what we need
3. GitHub's self-hosted runner setup is straightforward
4. Can use spot instances for ~70% cost savings

### Required Changes
1. Create Runner VM (modify existing script)
2. Update workflow files (minimal)
3. Handle runner lifecycle

## Cost Comparison
- **GitHub hosted**: Free (but killed at ~17min)
- **Self-hosted spot (persistent)**: ~$180/month running, ~$8/month stopped
- **Self-hosted spot (ephemeral)**: Only pay when jobs run (~$6/hour)

## Option 1: Quick Fix - Persistent Runner

### Setup Time: 1-2 hours

### What It Is
- Single VM runs GitHub Actions runner continuously
- Runner waits for jobs from GitHub
- Manually start/stop VM as needed

### Pros
- Simple to set up
- Predictable behavior
- Easy to debug

### Cons
- Costs $180/month when running
- Must manually manage VM lifecycle
- Single point of failure

### Implementation Steps
1. Create VM with GitHub runner installed
2. Register runner with repository
3. Update workflow files to use `runs-on: self-hosted`
4. Test with PR

## Option 2: Optimal - Ephemeral Runners

### Setup Time: 3-4 hours initial, more for edge cases

### What It Is
- VM spins up automatically when PR created/updated
- Runs single job then terminates
- Only pay for actual compute time

### Pros
- Lowest cost (only pay when running)
- Always fresh environment
- Scales automatically

### Cons
- More complex setup
- 2-3 minute VM startup delay
- Requires webhook handling
- Token management complexity

### Implementation Steps
1. Create VM startup script with runner installation
2. Set up webhook receiver (Cloud Function or Cloud Run)
3. Configure GitHub webhook to trigger on PR events
4. Update workflows to signal webhook
5. Handle cleanup and edge cases

### Architecture
```
GitHub PR Event
    ↓
GitHub Webhook
    ↓
GCP Cloud Function/Run
    ↓
Create Compute Instance (spot)
    ↓
Install & Register Runner
    ↓
Run GitHub Actions Job
    ↓
Auto-terminate VM
```

## Hybrid Approach (RECOMMENDED START)

### Setup Time: 30-45 minutes

Start with persistent runner to unblock immediately, then iterate toward ephemeral.

### Phase 1: Persistent Runner (NOW)
1. Create persistent self-hosted runner VM
2. Update workflow to `runs-on: self-hosted`
3. Test with current PR
4. Validate it solves the memory issue

### Phase 2: Iterate to Ephemeral (LATER)
1. Build webhook handler
2. Add VM lifecycle automation
3. Migrate to ephemeral model
4. Handle edge cases and monitoring

### Cost During Transition
- Persistent runner: ~$180/month when running, stop when not developing
- Same as Jupyter instance management
- Move to ephemeral once validated

## Key Files to Modify

### Workflows
- `.github/workflows/reusable_test.yaml:29` - change `runs-on: ubuntu-latest` to `runs-on: self-hosted`

### New Scripts
- `create_github_runner.sh` (based on `create_jupyter_spot.sh`)
- Optional: webhook handler for ephemeral runners

## Next Steps

**Immediate (Quick Win)**:
1. Create persistent runner VM script
2. Update workflow file
3. Test with PR #443

**Future Optimization**:
1. Implement webhook handler
2. Add ephemeral runner logic
3. Set up monitoring/alerts
4. Handle concurrent PRs
5. Cost optimization

## Token Management

### Persistent Runner
- Generate runner token once during setup
- Token expires, need to refresh periodically

### Ephemeral Runner
- Generate new token for each VM
- Use GitHub App or PAT with repo scope
- Tokens automatically cleaned up with VM

## Monitoring & Maintenance

### Things to Monitor
- VM costs (GCP billing alerts)
- Runner connectivity
- Job success rates
- VM cleanup (prevent zombie instances)

### Maintenance Tasks
- Token refresh (persistent)
- VM image updates
- Runner software updates
- Cost review monthly
