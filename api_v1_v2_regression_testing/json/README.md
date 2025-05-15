# API v1 and v2 regression testing JSON

## feature_http_requests.json

A formatted JSON list of policies, each of which corresponds with a feature requirement in a doc prepared by Nikhil [here](https://docs.google.com/document/d/1S1edqwPEvUuRisLHD0ih3hi9F-hQMRsnRpSEVVMBm1M/edit?tab=t.0#heading=h.cxuruc7p8lyx). The doc outlines which request object corresponds with which feature, as JSON does not permit comments.

## unique_http_requests.json

A formatted JSON list of all society-wide simulation requests to the PolicyEngine API from April 8 to May 7, 2025. This includes invalid requests and should not be run against the service without sanitization.

## unique_http_requests_filtered.json

`unique_http_requests.json`, filtered to remove invalid requests. 
