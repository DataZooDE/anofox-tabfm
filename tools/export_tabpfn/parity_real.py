"""Real-weight parity: download TabPFN v2 clf ckpt (transient), export a graph
WITH the real weights, run ORT vs PyTorch on the patched model. Weights never
committed. Prints argmax agreement + max abs logit diff on test rows."""
import pathlib, tempfile, sys
import numpy as np
import torch
from export_tabpfn import export
from export_tabpfn.tabpfn_patched import load_real_model, ExportWrapper

try:
    from tabpfn.model_loading import download_model, ModelVersion
    tmp = pathlib.Path(tempfile.mkdtemp())
    print("downloading real TabPFN v2 classifier ckpt (transient) ...", flush=True)
    download_model(to=tmp / "tabpfn-v2-classifier.ckpt", version=ModelVersion.V2,
                   which="classifier", model_name="tabpfn-v2-classifier.ckpt")
    ckpt = next(tmp.rglob("*.ckpt"))
    print("ckpt:", ckpt, ckpt.stat().st_size, flush=True)
    model = load_real_model("classification", str(ckpt))
    print("loaded real model: n_out", model.n_out, "params",
          sum(p.numel() for p in model.parameters()), flush=True)
    gp = tmp / "graph_real_clf.onnx"
    export.export_graph(model, gp, example=(16, 6, 10), max_classes=model.n_out)
    import onnxruntime as ort
    sess = ort.InferenceSession(str(gp), providers=["CPUExecutionProvider"])
    w = ExportWrapper(model).eval()
    rng = np.random.default_rng(7)
    worst, argok = 0.0, True
    for (T, H, N) in [(50, 8, 30), (100, 15, 60), (30, 5, 20)]:
        x = rng.standard_normal((1, T, H)).astype(np.float32)
        y = rng.integers(0, 4, (1, N)).astype(np.float32)
        (r,) = sess.run(["logits"], {"x": x, "y": y})
        with torch.no_grad():
            pt = w(torch.from_numpy(x), torch.from_numpy(y)).numpy()
        d = float(np.abs(r[:, N:] - pt[:, N:]).max())
        a = bool((r[:, N:].argmax(-1) == pt[:, N:].argmax(-1)).all())
        worst = max(worst, d); argok = argok and a
        print(f"  T={T} H={H} N={N}: max abs logit diff {d:.2e}, argmax agree {a}")
    print(f"REAL-WEIGHT PARITY: worst {worst:.2e}, all argmax agree {argok}")
    import shutil; shutil.rmtree(tmp, ignore_errors=True)
except Exception as e:
    import traceback; traceback.print_exc()
    print("REAL-WEIGHT PARITY UNAVAILABLE:", type(e).__name__, str(e)[:300])
    sys.exit(3)
