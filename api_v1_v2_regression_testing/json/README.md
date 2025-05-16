# API v1 and v2 regression testing JSON

## feature_http_requests.json

A formatted JSON list of policies, each of which corresponds with a feature requirement in a doc prepared by Nikhil [here](https://docs.google.com/document/d/1S1edqwPEvUuRisLHD0ih3hi9F-hQMRsnRpSEVVMBm1M/edit?tab=t.0#heading=h.cxuruc7p8lyx). The doc outlines which request object corresponds with which feature, as JSON does not permit comments.

## message_metrics.json

Accompanies `simulation_logs.json` below; divides all logs up by their output message. Logs with "CALCULATE_ECONOMY_SIMULATION_JOB: APIv2 job failed" mean that API v1 succeeded, while v2 failed. Logs containing "CALCULATE_ECONOMY_SIMULATION_JOB: APIv1 job failed" mean that v1 failed, regardless of v2's status.

## simulation_logs.json

All v1/v2 regression testing logs from 9pm Eastern Wednesday, May 14, to 8:15pm Eastern Thursday, May 15.

## unique_http_requests.json

A formatted JSON list of all society-wide simulation requests to the PolicyEngine API from April 8 to May 7, 2025. This includes invalid requests and should not be run against the service without sanitization.

## unique_http_requests_filtered.json

`unique_http_requests.json`, filtered to remove invalid requests. 
