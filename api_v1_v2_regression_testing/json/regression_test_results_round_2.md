Total regression tests: 36
* v1 and v2 complete and compare successfully: 26
* v1 fails: 1
  * Cause: nyc calculations not properly encouding using county FIPS: 1
    * Record: The only nyc policy in the set of 36 tests
* False-positive v2 fails: 1
  * Cause: UK log is too large
    * Sim API run: a2960ee9-8d59-4aef-912c-70f453383eea
* v2 fails: 6
  * Out of memory due to LSRs: 3
    * Sim API run: aa958644-7bc5-4529-ad48-a0bc412affc3
    * Sim API run: 2b44fe3e-e21c-4a96-821d-9d9f858c350b
    * Sim API run: f084c75f-9339-4df3-a017-aa627c2001f0
  * Unclear issue
    * Error: `api_v2_output["model_version"] - NoneType object is not subscriptable`
    * Sim API run: 9ffa1a1f-e4a9-480d-b83e-c61194ebb4df
    * Sim API run: eaae9e5f-9d1f-4187-809c-4109884c1885
    * Sim API run: cc0af437-525f-4a40-9942-82953db870d1
* Unclear: 2
  * For both, I get accurate results from API v1, but no job is ever initiated in v2
  * Policy: UK reform 81627 over baseline 1, region "uk", 2025
  * Poilcy: UK reform 84090 over baseline 1, region "uk", 2028

