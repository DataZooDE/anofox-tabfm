"""The masked export wrapper: the thing torch.onnx.export is handed.

Same routing as `tabpfn_patched.ExportWrapper`, because the engine surfaces the
graph's context rows as `is_training` fitted values and the CPU and GPU answers
have to agree on EVERY row, not just the query rows:

  * classification on v2.6 / v3 -- the model's own train-embedding decode,
    which is measurably right there;
  * v2, v2.5, and regression on every architecture -- the rows evaluated a
    second time as queries (see ExportWrapper._all_row_logits for the numbers).

Inputs are all padded to the shape bucket: x[1,T,H], y[1,T], and three int64
scalars -- train_size, n_rows (real rows) and d (real columns).
"""

from __future__ import annotations

import torch

import tabpfn.architectures.tabpfn_v2 as v2mod
import tabpfn.architectures.tabpfn_v2_5 as v25mod

from export_tabpfn.tabpfn_mask_forward import masked_forward
from export_tabpfn.tabpfn_patched import ExportWrapper


class MaskExportWrapper(ExportWrapper):
    """ExportWrapper with the split, the real row count and d as inputs."""

    @property
    def _duplicate(self) -> bool:
        return isinstance(self.m, (v2mod.TabPFNV2, v25mod.TabPFNV2p5)) or self.task == "regression"

    def forward(self, x, y, train_size, n_rows, d):
        xt = x.permute(1, 0, 2)                       # [T, 1, H]
        y_R = torch.nan_to_num(y[0], nan=0.0, posinf=0.0, neginf=0.0)
        T = xt.shape[0]
        ts = train_size.reshape(())
        nr = n_rows.reshape(())
        dd = d.reshape(())

        if self.task == "regression":
            train = torch.arange(T, device=xt.device) < ts
            n = ts.to(y_R.dtype)
            zero = torch.zeros_like(y_R)
            ymean = torch.where(train, y_R, zero).sum() / n
            # population std over the TRAIN rows, matching the fit path
            ystd = torch.clamp(torch.sqrt(torch.where(train, (y_R - ymean) ** 2, zero).sum() / n),
                               min=1e-20)
            logits = masked_forward(self.m, xt, (y_R - ymean) / ystd, ts, nr, dd,
                                    duplicate=self._duplicate)
            raw = self._bardist_mean(logits) * ystd + ymean
            full = raw.unsqueeze(-1)
        else:
            full = masked_forward(self.m, xt, y_R, ts, nr, dd, duplicate=self._duplicate)
        return full.permute(1, 0, 2)                  # [1, T, C]
