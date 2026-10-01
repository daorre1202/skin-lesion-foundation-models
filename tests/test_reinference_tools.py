import json

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import balanced_accuracy_score

from scripts import analyze_reinference as an
from scripts import ce1_common as cc


def test_ordered_ids_class_major_and_sorted():
    labels = {"ISIC_3": "NV", "ISIC_1": "MEL", "ISIC_2": "MEL", "ISIC_4": "BCC", "ISIC_5": "MEL"}
    assignment = {"ISIC_3": "test", "ISIC_1": "test", "ISIC_2": "test", "ISIC_4": "test", "ISIC_5": "val"}
    assert cc.ordered_ids(labels, assignment, "test") == ["ISIC_1", "ISIC_2", "ISIC_3", "ISIC_4"]
    assert cc.ordered_ids(labels, assignment, "val") == ["ISIC_5"]
    assert cc.label_array(labels, ["ISIC_1", "ISIC_3", "ISIC_4"]).tolist() == [0, 1, 2]


def test_load_labels_from_one_hot(tmp_path):
    for name, rows in {"train": [("A", "MEL")], "val": [("B", "NV")], "test": [("C", "VASC")]}.items():
        df = pd.DataFrame({"image": [r[0] for r in rows], **{c: [float(c == r[1]) for r in rows] for c in cc.CLASSES}})
        df.to_csv(tmp_path / cc.GT_FILES[name], index=False)
    assert cc.load_labels(tmp_path) == {"A": "MEL", "B": "NV", "C": "VASC"}


def test_index_images_first_root_wins(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b" / "sub"
    a.mkdir(), b.mkdir(parents=True)
    (a / "ISIC_1.jpg").write_bytes(b"x")
    (b / "ISIC_1.jpg").write_bytes(b"y")
    (b / "ISIC_2.JPG").write_bytes(b"z")
    found = cc.index_images([a, tmp_path / "b"])
    assert found["ISIC_1"].parent == a and set(found) == {"ISIC_1", "ISIC_2"}


def test_weighted_ensemble_is_normalised():
    p = {"a": np.array([[1.0, 0.0]]), "b": np.array([[0.0, 1.0]])}
    out = cc.weighted_ensemble(["a", "b"], p, {"a": 3.0, "b": 1.0})
    assert out.tolist() == [[0.75, 0.25]]


def test_bootstrap_matches_sklearn_on_the_full_sample():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 7, 400)
    pred = np.where(rng.random(400) < 0.7, y, rng.integers(0, 7, 400))
    boot = cc.paired_bootstrap(y, {"m": pred}, n_boot=2000, seed=1)["m"]
    point = balanced_accuracy_score(y, pred)
    assert abs(boot.mean() - point) < 0.01
    assert boot.min() < point < boot.max()


def test_bootstrap_is_paired_and_reproducible():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 7, 300)
    pred = np.where(rng.random(300) < 0.8, y, rng.integers(0, 7, 300))
    a = cc.paired_bootstrap(y, {"x": pred, "y": pred.copy()}, n_boot=500, seed=7)
    assert (a["x"] == a["y"]).all()
    c = cc.contrast(a, "x", "y", 0.8, 0.8)
    assert c["delta"] == 0 and c["ci_low"] == 0 and c["ci_high"] == 0 and c["p_positive"] == 0
    b = cc.paired_bootstrap(y, {"x": pred}, n_boot=500, seed=7)
    assert (a["x"] == b["x"]).all()


@pytest.fixture()
def fake_run(tmp_path):
    """Synthetic probabilities built from the canonical labels, to exercise the whole analysis."""
    yt, yv = np.load(cc.RUN / "tta_labels.npy"), np.load(cc.RUN / "tta_val_labels.npy")
    rng = np.random.default_rng(5)

    def probs(y, noise):
        logits = rng.normal(size=(len(y), 7)) * noise
        logits[np.arange(len(y)), y] += 2.0
        e = np.exp(logits)
        return (e / e.sum(1, keepdims=True)).astype(np.float32)

    for i, m in enumerate(cc.MODELS):
        for split, y in (("val", yv), ("test", yt)):
            for mode in ("raw", "tta"):
                np.save(tmp_path / f"probs_{m}_{split}_{mode}.npy", probs(y, 1.2 + 0.1 * i))
    return tmp_path


def test_analysis_runs_end_to_end(fake_run):
    r = an.analyze(fake_run, n_boot=300)
    assert (fake_run / "analysis.json").exists()
    assert set(r["contrasts"]) == {"all_tta_minus_cnn_tta", "foundation_tta_minus_cnn_tta",
                                   "dinov2_tta_minus_raw", "all_tta_minus_all_raw"}
    for c in r["contrasts"].values():
        assert c["ci_low"] <= c["ci_high"] and 0 <= c["p_positive"] <= 1
    s = r["subset_search"]
    assert s["n_subsets"] == 63 and 1 <= s["six_model_rank_on_val"] <= 63
    assert set(s["best_subset"]) <= set(cc.MODELS)
    assert json.loads((fake_run / "analysis.json").read_text())["n_boot"] == 300
