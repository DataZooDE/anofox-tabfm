"""Real-weight parity: the export forward vs upstream TabPFNV3p5.forward.

    uv run python parity_real.py <checkpoint.safetensors> [--onnx DIR]

Loads a released checkpoint through upstream's own loader, then for both tasks
compares, on EVERY row, `Wrapper35` (what the graphs are exported from) with the
same computation done by upstream's forward (`upstream_wrapped`). With --onnx it
also runs the exported positional graphs under ORT with the real weights
injected, which is the engine's CPU path minus the C++.

It also measures the fitted-values route on separable data, because that choice
was made by measurement for v3 and must not be assumed for 3.5: classification
fitted accuracy through the embedding route (what the graph does) and through
the duplicate route (rows presented again as queries).

Weights are read transiently from the user's own download and never written.
"""

from __future__ import annotations

import json
import sys
import time

import numpy as np
import torch

from export_tabpfn35.model import load_real_model
from export_tabpfn35.wrappers import Wrapper35, upstream_wrapped


def _feeds(task, seed=3):
    rng = np.random.default_rng(seed)
    out = []
    for T, H, N in ((60, 5, 40), (130, 9, 100), (33, 2, 20)):
        x = rng.standard_normal((1, T, H)).astype(np.float32)
        x[:, :, 0] = rng.integers(0, 4, (1, T))
        if task == "regression":
            y = (x[0, :N, 1] * 2.0 + x[0, :N, 0] + 0.1 * rng.standard_normal(N) + 5.0)[None]
        else:
            y = (x[0, :N, 1] > 0).astype(np.float32)[None] + (x[0, :N, 0] > 1)
        out.append((torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))))
    return out


def _separable(seed=0):
    rng = np.random.default_rng(seed)
    centres = np.array([[0, 0, 0, 0], [4, 4, 0, 0], [0, 4, 4, 0]], np.float32)
    lab = np.repeat(np.arange(3), 50)
    x = centres[lab] + 0.6 * rng.standard_normal((150, 4)).astype(np.float32)
    perm = rng.permutation(150)
    return torch.from_numpy(x[perm][None]), torch.from_numpy(lab[perm].astype(np.float32))


def main(argv=None) -> int:
    argv = list(argv if argv is not None else sys.argv[1:])
    ckpt = argv[0]
    onnx_dir = argv[argv.index("--onnx") + 1] if "--onnx" in argv else None
    torch.set_grad_enabled(False)
    report = {"checkpoint": ckpt}
    for task in ("classification", "regression"):
        model = load_real_model(ckpt, task)
        w = Wrapper35(model, task=task).eval()
        rows = []
        for x, y in _feeds(task):
            t0 = time.time()
            got = w(x, y)
            t_ours = time.time() - t0
            t0 = time.time()
            want = upstream_wrapped(model, x, y, task)
            t_up = time.time() - t0
            r = {"T": x.shape[1], "H": x.shape[2], "N": y.shape[1],
                 "max_abs": float((got - want).abs().max()),
                 "s_export_forward": round(t_ours, 3), "s_upstream": round(t_up, 3)}
            if task == "classification":
                r["argmax_agree"] = bool((got.argmax(-1) == want.argmax(-1)).all())
            rows.append(r)
        report[task] = rows
        print(task, json.dumps(rows), flush=True)

        if onnx_dir:
            from safetensors.torch import load_file
            stem = "tabpfn35fast" if model.icl_blocks.__len__() == 8 else "tabpfn35"
            g = f"{onnx_dir}/graph_{stem}_{task}.onnx"
            tmap = json.load(open(f"{onnx_dir}/tensor_map_{stem}_{task}.json"))["initializers"]
            from export_tabpfn35.cli import injected_session
            sess = injected_session(g, tmap, load_file(ckpt))
            worst = 0.0
            for x, y in _feeds(task):
                (o,) = sess.run(["logits"], {"x": x.numpy(), "y": y.numpy()})
                worst = max(worst, float(np.abs(o - w(x, y).numpy()).max()))
            report[f"{task}_onnx_vs_torch"] = worst
            print(task, "ORT (real weights) vs torch", worst, flush=True)

        if task == "classification":
            x, lab = _separable()
            N = 100
            y = lab[None, :N]
            full = w(x, y)[0].argmax(-1)
            emb_fit = float((full[:N] == lab[:N]).float().mean())
            qry = float((full[N:] == lab[N:]).float().mean())
            xt = x.permute(1, 0, 2)
            dup = model(torch.cat([xt[:N], xt[:N]], 0), lab[:N], "multiclass")[:, 0].argmax(-1)
            dup_fit = float((dup == lab[:N]).float().mean())
            report["fitted_route"] = {"query_acc": qry, "embedding_route_fitted_acc": emb_fit,
                                      "duplicate_route_fitted_acc": dup_fit}
            print("fitted-values route", report["fitted_route"], flush=True)
        else:
            # Regression: how good the CHEAP route would have been, for the record.
            x, _ = _separable()
            tgt = (x[0, :, 0] * 1.5 - x[0, :, 1] + 2.0)
            N = 100
            y = tgt[None, :N]
            dup_fit = w(x, y)[0, :N, 0]
            corr = float(np.corrcoef(dup_fit.numpy(), tgt[:N].numpy())[0, 1])
            ym, ys = y.mean(), y.std(correction=0)
            xt = x.permute(1, 0, 2)
            res = model(xt, (y[0] - ym) / ys, "regression", only_return_standard_out=False)
            tr = res["train_embeddings"].transpose(0, 1)
            emb_logits = model.heads(None, tr, None, task_type="regression", num_present_classes=None)
            emb_fit = w._bardist_mean(emb_logits)[:, 0] * ys + ym
            corr_emb = float(np.corrcoef(emb_fit.numpy(), tgt[:N].numpy())[0, 1])
            report["fitted_route_regression"] = {"duplicate_route_corr": corr,
                                                 "embedding_route_corr": corr_emb}
            print("regression fitted route", report["fitted_route_regression"], flush=True)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
