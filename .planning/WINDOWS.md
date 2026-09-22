---
schema_version: 1
open_count: 0
waived_count: 0
fixed_count: 4
total_count: 4
last_updated: 2026-09-22T18:16:41.705Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 1 | stub | src/tabfm_metrics_regression.cpp |  | RegisterRegressionMetrics body is a stub — registers nothing; filled by plan 01-03 | fixed |  | 2026-09-20T20:45:27.517Z | 2026-09-22T18:16:31.897Z |
| 2 | 1 | stub | src/tabfm_crossval.cpp |  | RegisterCrossValidateMacros body is a stub — registers nothing; filled by plan 01-04 | fixed |  | 2026-09-20T20:45:27.643Z | 2026-09-22T18:16:41.377Z |
| 3 | 02 | stub | src/tabfm_preprocess_tabpfn_v2.cpp |  | TabPFNV2PreprocessBatch throws NotImplementedException — fixture-scoped implementation deferred to plan 02-03 | fixed |  | 2026-09-21T19:56:14.564Z | 2026-09-22T18:16:41.562Z |
| 4 | 02 | stub | test/cpp/test_tabfm_distribution_decode.cpp |  | Hidden placeholder [.] test case — plan 02-02 fills with distribution decode golden tests | fixed |  | 2026-09-21T19:56:14.691Z | 2026-09-22T18:16:41.705Z |

````json
[
  {
    "id": 1,
    "kind": "stub",
    "phase": "1",
    "file": "src/tabfm_metrics_regression.cpp",
    "line": null,
    "description": "RegisterRegressionMetrics body is a stub — registers nothing; filled by plan 01-03",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-20T20:45:27.517Z",
    "resolved_at": "2026-09-22T18:16:31.897Z"
  },
  {
    "id": 2,
    "kind": "stub",
    "phase": "1",
    "file": "src/tabfm_crossval.cpp",
    "line": null,
    "description": "RegisterCrossValidateMacros body is a stub — registers nothing; filled by plan 01-04",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-20T20:45:27.643Z",
    "resolved_at": "2026-09-22T18:16:41.377Z"
  },
  {
    "id": 3,
    "kind": "stub",
    "phase": "02",
    "file": "src/tabfm_preprocess_tabpfn_v2.cpp",
    "line": null,
    "description": "TabPFNV2PreprocessBatch throws NotImplementedException — fixture-scoped implementation deferred to plan 02-03",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T19:56:14.564Z",
    "resolved_at": "2026-09-22T18:16:41.562Z"
  },
  {
    "id": 4,
    "kind": "stub",
    "phase": "02",
    "file": "test/cpp/test_tabfm_distribution_decode.cpp",
    "line": null,
    "description": "Hidden placeholder [.] test case — plan 02-02 fills with distribution decode golden tests",
    "status": "fixed",
    "reason": "",
    "recorded_at": "2026-09-21T19:56:14.691Z",
    "resolved_at": "2026-09-22T18:16:41.705Z"
  }
]
````
