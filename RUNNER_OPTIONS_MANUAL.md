
  Yes, this is absolutely possible! Here's the approach:

  The Setup

  1. Create a manual workflow (.github/workflows/manual-test-runner.yml) that:
  - Starts the VM
  - Schedules auto-shutdown in X hours
  - Tests run automatically on the self-hosted runner

  2. Make PR tests require this manual step:
  - Change Quick Feedback to use self-hosted ✅ (already done)
  - Add this new manual workflow as a required status check
  - PRs can't merge until someone clicks the button

  How It Works

  1. Developer opens/updates PR → Quick Feedback job sits in "waiting for runner" state
  2. Reviewer clicks "Run workflow" button in Actions tab → enters PR number
  3. Workflow starts VM → schedules shutdown in 4 hours → tests run automatically
  4. Tests complete → VM stays up for other PRs to use (up to 4hr limit)
  5. After 4 hours → VM auto-shuts down, saving costs

  The Workflow File

  Here's what you'd create as .github/workflows/manual-test-runner.yml:

  name: Start Test Runner (Self-Hosted)
  on:
    workflow_dispatch:
      inputs:
        shutdown_hours:
          description: 'Auto-shutdown after (hours)'
          required: false
          default: '4'

  jobs:
    start-runner:
      runs-on: ubuntu-latest
      steps:
        - name: Authenticate to GCP
          uses: google-github-actions/auth@v2
          with:
            credentials_json: ${{ secrets.GCP_SA_KEY }}

        - name: Set up Cloud SDK
          uses: google-github-actions/setup-gcloud@v2

        - name: Start Runner VM
          run: |
            gcloud compute instances start github-runner --zone=us-central1-a
            sleep 30  # Wait for boot

        - name: Schedule Auto-Shutdown
          run: |
            MINS=$((${{ inputs.shutdown_hours }} * 60))
            gcloud compute ssh github-runner --zone=us-central1-a --command \
              "echo 'sudo shutdown -h now' | at now + ${MINS} minutes"

        - name: Report
          run: |
            echo "✅ Runner started, will shutdown in ${{ inputs.shutdown_hours }}h"
            echo "Tests for waiting PRs will now run automatically"

  To make this required: Add it to branch protection rules so PRs can't merge until someone runs it.

  Want me to create this file for you? I'd need the GCP service account key added to GitHub secrets first (GCP_SA_KEY).

