Total regression tests: 36
* v1 and v2 complete and compare successfully: 26
  * v1 and v2 budgetary impacts differ by 5% or less: 18
  * v1 and v2 budgetary impacts differ by 5%+: 8 (sorted by difference below)
    * Sim API run: 07874967-79e4-4f82-af7d-4b0231fc4941; difference: 2,304%
    * Sim API run: ab8bc6c8-6956-4a0e-ac07-1d31c00fd075; difference: 18%
    * Sim API run: 51cc31b3-fd2d-4605-8819-178d59458926
    * Sim API run: 98a82089-745a-4530-803b-421d7f896077
    * Sim API run: 51bcf9b0-d0be-417f-8b6f-4595b939eb04
    * Sim API run: 4fce5bce-fe06-404b-8453-3d8a6525e07d
    * Sim API run: 3b1642c7-2aed-4f3a-adfa-56076f258eb2
    * Sim API run: 7972afa7-25c3-4761-937b-c18a048c3c72
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

