# API v1 and v2 regression testing scripts

## emit_structured_requests.py

The `emit_structured_requests.py` script is a Python tool for batch processing economic impact requests to the PolicyEngine API. It allows you to sequentially send multiple simulation requests while managing rate limits and polling for results.

### Usage

1. Create a JSON file containing an array of request objects (default: `policy_requests.json`). See a sample schema below.
2. Ensure the `regression_runs` folder exists in your local directory before running the script.
3. Run the script: `python emit_structured_requests.py [custom_input_file.json]`
4. Results will be saved to a timestamped file in the `regression_runs` folder.

### Request Object Schema

Each request in the JSON array should follow this structure:

```json
{
 "reform": 123,               // Required: ID of the reform policy
 "baseline": 456,             // Required: ID of the baseline policy
 "region": "us",              // Optional: Geographic region (default: "us")
 "timePeriod": 2025,          // Required: Year for the simulation
 "dataset": "cps",            // Optional: Dataset to use for simulation
 "maxHouseholds": 1000,       // Optional: Maximum number of households to include
 "household": 42,             // Optional: Specific household ID to simulate
 "mode": "normal"             // Optional: Simulation mode
}
```

### Important notes
- The script requires the `requests` library (`pip install requests`)
- Request batches are processed with pauses to avoid overloading the API
- The default configuration polls up to 600 times with 1-second intervals, allowing a simulation run to take up to 10 minutes before timing out
- Requests are processed in batches of 10 with 2-minute pauses between batches

# extract_http_requests.py

The `extract_http_requests.py` script is a Python tool for extracting HTTP PolicyEngine API request logs from GCP. It implements rate limiting to avoid GCP throttling.

## Usage

1. Ensure you have appropriate GCP credentials configured in your environment.
2. Install required dependencies: `pip install google-cloud-logging`
3. Run the script with desired parameters:

  ```bash
  python extract_http_requests.py [options]
  ```

## Command Line Options
--project-id      GCP project ID (default: "policyengine-app")
--start-time      Start time in ISO format (e.g., 2025-04-01T00:00:00+00:00)
--end-time        End time in ISO format (e.g., 2025-05-01T00:00:00+00:00)
--output-file     Path to save results (default: "http_requests.json")
--max-results     Maximum number of results to return (default: 1,000,000,000)
--page-size       Number of results per API call (default: 100)
--sleep-interval  Records to process before sleeping (default: 100)
--sleep-duration  Sleep duration in seconds (default: 1.0)

## Important Notes

* This script requires the `google-cloud-logging` library
* Only requests with paths starting with "/us/policy" or "/uk/policy" are extracted
* The script processes logs in 1-hour time windows to avoid memory issues
* Results are saved in a simplified JSON format with resource path and timestamp
* If not specified, the time range defaults to the last 24 hours
* The script implements pauses to avoid rate limiting on the Google Cloud API
## pull_all_logs.py

This script searches through PolicyEngine's Google Cloud logs, finds all logs matching a certain condition, then batches them and provides analytics. It is still under construction and has not yet been tested. More info to come later.
