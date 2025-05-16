# API v1 passes, v2 fails

## V2 can't handle baseline-baseline comparison

- 97ab74a4-2f51-4c74-b278-c2b012e89ed1
- bde5385d-6896-4a9f-982b-1de581f6c89a

## Enhanced_us region subsampling returns empty dataset, causing simulation build to fail with NoneType (this shouldn't exist if region is being set properly by web app)

- ec4bdad8-71ba-4b26-b6fd-c1d591d27e42
- f27ac2df-4b49-49a2-9d61-180bde0ce2e1
- 0e8d638b-06ae-4244-b257-a7751c6063c5
- 7fc88e8c-8cff-4084-9017-a6b0aba1b6d6
- f19d047a-0438-420b-9928-e6a37cd0fe12
- c8764d96-3c80-4e87-b4f6-2a1c74d13ed7
- f76d1c49-0e21-46bb-9d62-ad387666a89f
- e9c2387e-6791-47ee-ba83-fb27062170c4
- d9fdde13-2a37-46f6-82be-aeda21f666ae
- 01344fff-c209-4e49-ac48-9e85d93653cb
- 59498ce3-669d-43e9-8c86-a93bc2a80f1a
- 527e3399-0375-45ac-b4f2-de66e19b692f
- 9237956e-3a8c-46b9-8a8a-fa8764c723fb
- c2b1cff3-b1d6-40de-8862-6cbb4caa98a4
- 73d62c6c-71ca-4558-9e43-5286152fb28c
- 2968de8b-44d8-4c94-808c-2de71a5c87c9
- 87cc676a-595a-4c8e-b453-ac5c9898de53
- 2572fc2c-e952-4ce4-a5de-a7d15bb701e8
- bc47877a-780e-46b9-b0b2-52ea6f05de40
- 90f901fc-7e86-4fd2-8135-146f8970bbb9
- 27bc0c18-170e-4fc0-9093-ba12ecb94a47
- ab9f17b5-3ea4-4f41-a771-e4915478b0de
- e19a86c2-5366-42d4-9fbb-3bf3a5d4ae23
- d36ea978-fb63-4bc0-879b-6c0faa9c3db9
- 651fdbef-2258-4577-a143-e73f05dfaf49

## Infinity input parsed as string, causing param setting to fail (this should be solved post-merge of #116 in .py)

- 08f7cbee-128b-4569-8414-a7197ddb07e4
- a8806090-9096-4165-a389-0b1d2bb8bd86
- 29305259-a61b-4173-b699-4fee6a7f16af
- 46856bd5-4a83-4441-9a5e-98d39001b60c

## Certain policies use too much memory, causing container to crash; eventually, we receive a timeout error

- 78c4ac80-4e30-40ad-8e6c-7ad4f9874d63 (contains LSRs)
- 69ec6e32-f6d4-4153-bce0-fbed6eb84a58 (UK LSRs with capital gains responses)
- 4bdb2c6c-1cbe-4883-8db1-d501817299d4 (UK LSRs with capital gains responses)
- a4e0ed18-a593-4a88-a417-0b72f77057c9 (TCJA with custom baseline; unclear why this would run out of memory)
- 6ecf47be-1e16-437b-9d54-9e3214ff6ecf (contains LSRs)
- 6f47477e-e3d4-4d80-8146-84609d2d1fcf (UK LSRs with capital gains responses)
- 93454af8-f98e-42dd-8cf6-b379f9590ec0 (US LSRs)

## Certain numeric parameters are sometimes encoded as strings, causing parameter loading failure

- 5fedaeea-bac2-4a6b-b2d8-d853b22dafec

## We specifically prevent running simulations prior to 2024, despite surfacing 2023 as an option

- ec1c1518-feae-4047-a284-7285d51ad12a
- c8e7dfb3-0218-4746-b907-fd4c1d8823a2
- d51d671e-2ac4-43ed-a439-fd6fbcb72255
- 990c80bf-c7ad-493a-a824-e79fb324e4c1

## Job passed, but log entry is too large, causing false failure

- 438f31e1-f1a0-4dd8-be1d-a970cb73777e
- cc3c2880-e38f-42a3-8317-53d83759e67f

## Job itself passes; unclear error

- 10087652-4643-46fb-849e-d59cce194b52

# API v1 fails

## Parameter missing in tax system

It appears that the logging is more straightforward when the param is basic, but when it's a bracket or other advanced param type, error is often opaque KeyError

- Reform 18192, region US, period 2023 (WIDOW) x2
- Reform 8952, region US, period 2024 (WIDOW)
- Reform 51191, region enhanced_us, period 2025 (WIDOW)
- Reform 24104, region US, period 2023 (WIDOW)
- Reform 51189, region enhanced_us, period 2025 (WIDOW)
- Reform 81993, region uk, period 2025 (weeks_per_year)
- Baseline 72299, reform 2, region MD, period 2024 (widow)
- Reform 57153, region enhanced_us, period 2024 (labor_supply)
- Reform 55168, region us, period 2024 (labor_supply)
- Reform 5902, region ma, period 2023 (dependent)
- Reform 44918, region enhanced_us, period 2024 (WIDOW)
- Reform 53757, region enhanced_us, period 2024 (labor_supply)
- Reform 2992, region uk, period 2023 (abolitions)
- Reform 44066, region us, period 2023 (ctc_expansion)
- Reform 3305, region uk, period 2023 (abolitions)
- Reform 47709, region enhanced_us, period 2024 (WIDOW)
- Reform 54413, region enhanced_us, period 2024 (labor_supply)

## Poverty impact data's invocation of microdf causes DivisionByZeroError

- Reform 5331, region CA, period 2023
- Reform 49861, region enhanced_us, period 2023
- Reform 6454, region us, period 2023
- Baseline 84117, reform 49667, period 2023
- Baseline 3318, reform 2, region pa, period 2023
- Reform 10720, region ut, period 2023
- Reform 40336, region md, period 2023
- Reform 14, region us, period 2023
- Reform 8918, region ks, period 2023
- Reform 10723, region ut, period 2023
- Reform 19600, region dc, period 2023
- Baseline 6525, reform 2, region us, period 2023
- Baseline 3318, reform 2, region us, period 2023
- Reform 6524, region us, period 2023
- Reform 15208, region us, period 2023
- Reform 3938, region us, period 2023
- Reform 28026, region us, period 2023
- Reform 28034, region us, period 2023
- Reform 28028, region us, period 2023
- Reform 40330, region md, period 2023
- Baseline 22000, reform 2, region nm, period 2023
- Baseline 63889, reform 2, region nm, period 2023
- Reform 32694, region enhanced_us, period 2023
- Reform 5828, region ma, period 2023
- Baseline 2, reform 2, region us, period 2023
- Reform 10613, region ut, period 2023
- Reform 49862, region enhanced_us, period 2023

## Potential error in formula for "county" in -us

- Reform 72622, region nyc, period 2029
- Reform 14, region nyc, period 2025
- Reform 72622, region nyc, period 2028
- Reform 74381, region nyc, period 2025 x3

File "/usr/local/lib/python3.10/site-packages/pandas/core/indexes/base.py", line 6130, in _raise_if_missing raise KeyError(f"None of [{key}] are in the [{axis_name}]") KeyError: "None of [Int64Index([ 5, 5, 5, 5, 5, 5, 5, 5, 5, 5,\n ...\n 85, 85, 85, 85, 85, 85, 85, 85, 85, 85],\n dtype='int64', name='county_fips', length=2954)] are in the [index]"

## No time period supplied

- Reform 76143, region ky, period None

## (Potentially) improperly programmed a standard param as a list param?

- Reform 43092, region enhanced_us, period 2024
- Reform 43149, region enhanced_us, period 2024
