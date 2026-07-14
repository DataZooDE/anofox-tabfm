"""Fast reproducibility + license-wall guard for the Mitra exporter.

Exports the tiny 'fixture' config end to end and asserts:
  * the exported graph exposes EXACTLY (x, y, train_size, d) -> logits
    (cat_mask omitted),
  * every checkpoint-mapped initializer is EXTERNAL (weight-free) after the
    .onnx.data is deleted,
  * ORT(weight-free graph + injected random weights) matches patched PyTorch on
    shapes different from the export example (tol 1e-3).
"""

import pathlib
import tempfile

import onnx

from export_mitra import configs, export


def _run(task):
    cfg = configs.get("fixture", task)
    with tempfile.TemporaryDirectory() as d:
        gp = pathlib.Path(d) / f"graph_mitra_{task}.onnx"
        model = export.build_model(task, cfg.model_kwargs, seed=0)
        wrapper = export.export_graph(
            model, gp, dim_rows=cfg.dim_rows, dim_features=cfg.dim_features,
            example=cfg.example)

        proto = onnx.load(str(gp), load_external_data=False)
        names = [i.name for i in proto.graph.input]
        assert names == ["x", "y", "train_size", "d"], names
        assert [o.name for o in proto.graph.output] == ["logits"]

        state = {k: v for k, v in model.state_dict().items()}
        tmap = export.postprocess(gp, state)
        assert not tmap["unmatched_small"]
        assert all(o == "m." + s for o, s in tmap["initializers"].items())

        parity = export.check_parity(gp, model, wrapper, cfg.parity_shapes)
        assert parity["ok"], parity

        export.delete_weight_data(gp)
        export.assert_weight_free(gp, tmap)  # raises if any weight byte remains


def test_classification():
    _run("classification")


def test_regression():
    _run("regression")
