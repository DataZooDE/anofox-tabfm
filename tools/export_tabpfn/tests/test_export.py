"""Smoke test: fixture-config export produces a general, weight-free graph
whose ORT run matches PyTorch (random weights) on shapes != the export example."""
import pathlib

from export_tabpfn import configs, export
from export_tabpfn.tabpfn_patched import build_random_model


def test_classification_fixture_export(tmp_path):
    cfg = configs.fixture()
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model("classification", kw, seed=0)

    graph = tmp_path / "g.onnx"
    export.export_graph(model, graph, example=cfg.example, max_classes=cfg.max_classes)
    tmap = export.postprocess(graph, dict(model.state_dict()))

    # 29 transformer/embedder/decoder weights map; _pos_base stays inline.
    assert len(tmap["initializers"]) == 29
    assert not tmap["unmatched_small"]
    assert "_pos_base" not in tmap["initializers"].values()

    parity = export.check_parity(graph, model, cfg.parity_shapes,
                                 max_classes=cfg.max_classes)
    assert parity["ok"], parity
    assert parity["argmax_all_agree"], parity

    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tmap)  # raises if any weight bytes on disk


def test_regression_fixture_export(tmp_path):
    cfg = configs.fixture()
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model("regression", kw, seed=0)
    assert model.n_out == cfg.num_buckets  # bar-distribution logits, not 1

    graph = tmp_path / "g.onnx"
    export.export_graph(model, graph, example=cfg.example, max_classes=cfg.num_buckets)
    tmap = export.postprocess(graph, dict(model.state_dict()))
    parity = export.check_parity(graph, model, cfg.parity_shapes,
                                 max_classes=cfg.num_buckets)
    assert parity["ok"], parity
