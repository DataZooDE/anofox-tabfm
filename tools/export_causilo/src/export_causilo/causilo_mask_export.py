"""The masked export wrapper: the thing torch.onnx.export is handed.

Same outputs as the positional `causilo_patches.ExportWrapper` (every row is a prediction, rows < train_size
are in-context fitted values, regression is a RAW-space point estimate), but the split, the real row count and
the real feature count arrive as VALUES, so the only dynamic axes are rows and features, the two the
(T, H) shape-bucket table already covers:

    x          [1, T, H] f32   real data in [:n_rows, :d], zeros elsewhere
    y          [1, T]    f32   train targets in [:train_size], -100 elsewhere
    train_size [1] int64
    n_rows     [1] int64       declared so the MIGraphX plugin binds it uniformly; Causilo does not need it
                               (nothing here looks at all rows; docs/ROCM_CAUSILO_PLAN.md)
    d          [1] int64
"""

from __future__ import annotations

import torch

from export_causilo.causilo_mask_forward import masked_forward


class MaskExportWrapper(torch.nn.Module):
    """Pins Causilo to the five-input masked ONNX signature."""

    def __init__(self, model):
        super().__init__()
        self.m = model

    def forward(self, x, y, train_size, n_rows, d):
        del n_rows
        # Indexing, NOT .reshape(()): a Reshape to an empty dims list exports as a Reshape whose target is []
        # and MIGraphX reads that as ZERO elements, rejecting the graph ("reshape has 0 elements whereas the
        # input has 1"). ORT accepts it, so only the GPU sees it.
        y_row = torch.nan_to_num(y[:, :], nan=0.0, posinf=0.0, neginf=0.0)
        return masked_forward(self.m, x, y_row, train_size[0], d[0])
