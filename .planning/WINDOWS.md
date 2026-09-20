---
schema_version: 1
open_count: 2
waived_count: 0
fixed_count: 0
total_count: 2
last_updated: 2026-09-20T20:45:27.643Z
---

# Broken Windows Ledger

> Cross-phase defect register. With `workflow.windows_enforce` enabled, `/gsd-ship` blocks while `open_count > 0`.
> Waive with `gsd-tools windows waive <id> "<reason>"` (reason required).
> Mark fixed with `gsd-tools windows fixed <id>`.

| id | phase | kind | file | line | description | status | reason | recorded_at | resolved_at |
|----|-------|------|------|------|-------------|--------|--------|-------------|-------------|
| 1 | 1 | stub | src/tabfm_metrics_regression.cpp |  | RegisterRegressionMetrics body is a stub — registers nothing; filled by plan 01-03 | open |  | 2026-09-20T20:45:27.517Z |  |
| 2 | 1 | stub | src/tabfm_crossval.cpp |  | RegisterCrossValidateMacros body is a stub — registers nothing; filled by plan 01-04 | open |  | 2026-09-20T20:45:27.643Z |  |

````json
[
  {
    "id": 1,
    "kind": "stub",
    "phase": "1",
    "file": "src/tabfm_metrics_regression.cpp",
    "line": null,
    "description": "RegisterRegressionMetrics body is a stub — registers nothing; filled by plan 01-03",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-20T20:45:27.517Z",
    "resolved_at": null
  },
  {
    "id": 2,
    "kind": "stub",
    "phase": "1",
    "file": "src/tabfm_crossval.cpp",
    "line": null,
    "description": "RegisterCrossValidateMacros body is a stub — registers nothing; filled by plan 01-04",
    "status": "open",
    "reason": "",
    "recorded_at": "2026-09-20T20:45:27.643Z",
    "resolved_at": null
  }
]
````
