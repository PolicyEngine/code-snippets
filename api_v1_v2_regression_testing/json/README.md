# API v1 and v2 regression testing markdown & JSON
This folder contains JSON and markdown files used as part of API v1/v2 regression testing. The "Outputs/diagnostic logs" header below contains the outputs of the testing, while "Inputs" includes all individual regression test policies.

## Outputs/diagnostic logs

### v1_v2_error_list.md

An exhaustive list of all cases where API v2 failed while API v1 succeeded, and where API v1 failed regardless of API v2 status. The list categorizes each run by failure cause, grouping v2 failures by Google Cloud Workflow run ID and listing relevant pieces of setup info for v1 failures.

### simulation_logs.json

All v1/v2 regression testing logs from 9pm Eastern Wednesday, May 14, to 8:15pm Eastern Thursday, May 15. This is a massive, exhaustive file of more than 1 million lines.

### message_metrics.json

Accompanies `simulation_logs.json`; divides all logs up by their output message. Logs with "CALCULATE_ECONOMY_SIMULATION_JOB: APIv2 job failed" mean that API v1 succeeded, while v2 failed. Logs containing "CALCULATE_ECONOMY_SIMULATION_JOB: APIv1 job failed" mean that v1 failed, regardless of v2's status.

### budgetary_impact_comparisons.json

Aggregates all regression test outputs where v1 and v2 both succeeded, but the two differed by 5% or more, ordered by the difference measured.

## Inputs

### feature_http_requests.json

A formatted JSON list of policies, each of which corresponds with a feature requirement in a doc prepared by Nikhil [here](https://docs.google.com/document/d/1S1edqwPEvUuRisLHD0ih3hi9F-hQMRsnRpSEVVMBm1M/edit?tab=t.0#heading=h.cxuruc7p8lyx). The doc outlines which request object corresponds with which feature, as JSON does not permit comments.


### unique_http_requests.json

A formatted JSON list of all society-wide simulation requests to the PolicyEngine API from April 8 to May 7, 2025. This includes invalid requests and should not be run against the service without sanitization.

### unique_http_requests_filtered.json

`unique_http_requests.json`, filtered to remove invalid requests. 
