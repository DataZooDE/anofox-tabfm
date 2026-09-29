"""Export each architecture's masked graph in a FRESH process.

`mask_parity_full` patches all three architecture modules up front, so it cannot
see a forward that only works when the module it reaches into was the one
patched. That is exactly what happened: the 2.5 branch called
`v26.select_features`, `apply_module_patches("v2.5")` patches v2.5's module, and a
v2.5-only export hit upstream's data-dependent `torch.all(sel)`. The gate passed;
the real flow failed. A subprocess per architecture is the only faithful way to
reproduce the real flow.
"""
import subprocess
import sys

import pytest

CASES = [("fixture", "v2"), ("fixture25", "v2.5"), ("fixture26", "v2.6")]


@pytest.mark.parametrize("config,arch", CASES)
@pytest.mark.parametrize("task", ["classification", "regression"])
def test_masked_export_in_a_fresh_process(tmp_path, config, arch, task):
    r = subprocess.run(
        [sys.executable, "-m", "export_tabpfn.cli", "--task", task, "--config", config,
         "--contract", "mask", "--skip-parity", "--out", str(tmp_path)],
        capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, f"{arch} {task} masked export failed in isolation:\n{r.stderr[-1500:]}"
