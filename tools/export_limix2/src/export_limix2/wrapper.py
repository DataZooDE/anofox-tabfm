"""ExportWrapper for LimiX-2 (SPIKE). Same (x, y)-only contract as the LimiX-2M exporter.

  x [1, T, H] float32   all rows, H dynamic;  y [1, S] float32 TRAINING targets only (S = len(y) <= T)
  classification -> [1, T, 10] class logits (engine applies temperature/softmax)
  regression     -> [1, T, 1]  RAW-space mean of the 5000-bucket distribution

Rows < S are in-context fitted values via the duplicate route: the model sees `[context ; all rows]`,
eval_pos = S (measured necessary for LimiX-2M and Causilo; re-measured for LimiX-2 in the spike report).
Regression: y is z-scored in-graph (ddof=1, v2 predictor.py:3182) and the bucket mean is inverted
(softmax(logits / TEMP) . centers) * std + mean. TEMP = upstream's default softmax_temperature (0.9).
"""
from __future__ import annotations

import torch

REG_TEMP = 0.9


class ExportWrapper2(torch.nn.Module):
    def __init__(self, model, task: str):
        super().__init__()
        assert task in ("classification", "regression")
        self.m, self.task = model, task
        if task == "regression":
            c = (model._reg_borders[:-1] + model._reg_borders[1:]) / 2.0
            self.register_buffer("centers", c.clone(), persistent=False)

    def forward(self, x, y):
        s = y.shape[1]
        torch._check(s >= 2)  # normalize_mean0_std1 branches on eval_pos == 1; ddof=1 needs S >= 2
        torch._check(s <= x.shape[1])  # stops Min(S, T) sympy blow-up in the 24 per-layer slices
        if self.task == "regression":
            mean_y = y.mean(dim=1, keepdim=True)
            std_y = ((y - mean_y) ** 2).sum(dim=1, keepdim=True).div(s - 1).sqrt()
            std_y = torch.where(std_y == 0, torch.ones_like(std_y), std_y)
            y_in = (y - mean_y) / std_y
        else:
            y_in = y
        table = torch.cat([x[:, :s], x], dim=1)
        # eval_pos = S, but the duplicate route needs S queries more than rows: eval_pos is the CONTEXT length
        out = self.m(x=table, y=y_in, eval_pos=s, task_type="Classification" if self.task == "classification" else "Regression")
        if self.task == "classification":
            return out["cls_output"]
        logits = out["reg_output"][0]  # [1, Q, 5000]
        p = torch.softmax(logits / REG_TEMP, dim=-1)
        v = (p * self.centers).sum(-1, keepdim=True)
        return v * std_y.unsqueeze(-1) + mean_y.unsqueeze(-1)
