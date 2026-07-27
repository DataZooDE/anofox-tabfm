"""Smoke test: fixture-config export produces a general, weight-free graph
whose ORT run matches PyTorch (random weights) on shapes != the export example."""
import pathlib

from export_tabpfn import configs, export
from export_tabpfn.tabpfn_patched import ExportWrapper, build_random_model


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


# --- TabPFN-2.5 ---------------------------------------------------------------

def test_v25_fixture_exports_and_parity(tmp_path):
    """The 2.5 architecture exports through the same pipeline as v2."""
    cfg = configs.fixture25()
    assert cfg.arch == "v2.5"
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model("classification", kw, seed=0, arch=cfg.arch)
    graph = tmp_path / "graph25.onnx"
    export.export_graph(model, graph, example=cfg.example,
                        max_classes=cfg.max_classes, task="classification")
    tmap = export.postprocess(graph, dict(model.state_dict()))
    parity = export.check_parity(graph, model, cfg.parity_shapes,
                                 max_classes=cfg.max_classes, task="classification")
    assert parity["ok"], parity
    export.delete_weight_data(graph)
    export.assert_weight_free(graph, tmap)


def test_v25_column_embeddings_are_not_checkpoint_weights(tmp_path):
    """`_pos_base` comes from `tabpfn` package data and must stay OUT of the map.

    It is a fixed cross-platform table, not a trained tensor — baking it inline
    keeps the graph self-contained without shipping any Prior Labs weight bytes.
    """
    cfg = configs.real25("classification")
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model("classification", kw, seed=0, arch=cfg.arch)
    assert "_pos_base" in dict(model.state_dict())
    graph = tmp_path / "graph25real.onnx"
    export.export_graph(model, graph, example=cfg.example,
                        max_classes=cfg.max_classes, task="classification")
    tmap = export.postprocess(graph, dict(model.state_dict()))
    assert "_pos_base" not in set(tmap["initializers"].values())


def test_v25_target_range_guard_is_skipped(tmp_path):
    """Out-of-range targets must not raise: the guard is pinned off for export."""
    import torch
    cfg = configs.fixture25()
    kw = dict(cfg.model_kwargs)
    kw["num_buckets"] = cfg.num_buckets
    kw["max_num_classes"] = cfg.max_classes
    model = build_random_model("classification", kw, seed=0, arch=cfg.arch)
    w = ExportWrapper(model, task="classification").eval()
    # y far outside 0..n_out-1 would trip TabPFNV2p5's ValueError in eval mode.
    with torch.no_grad():
        out = w(torch.randn(1, 20, 5), torch.full((1, 12), 99.0))
    assert out.shape == (1, 20, cfg.max_classes)
