import json

import numpy as np
import pandas as pd
import pytest

from scripts import analyze_reinference as an
from scripts import ce1_common as cc

FOLDER = cc.ROOT / "results" / "reinference_2026-10"
README = (cc.ROOT / "README.md").read_text(encoding="utf-8")
META = json.loads((FOLDER / "reinference_meta.json").read_text())
STORED = json.loads((FOLDER / "analysis.json").read_text())


def f4(x):
    return f"{x:.4f}"


def signed(x):
    return f"{x:+.4f}"


@pytest.fixture(scope="module")
def probs():
    return an.load_probs(FOLDER)


def test_files_and_shapes(probs):
    assert len(list(FOLDER.glob("probs_*.npy"))) == 24
    for (split, _), per_model in probs.items():
        for m, arr in per_model.items():
            assert arr.shape == ((2342 if split == "val" else 2348), 7), (m, split)
            assert np.allclose(arr.sum(1), 1, atol=1e-4)


def test_run_settings_and_status():
    assert META["tta_rounds"] == 10 and META["tta_seed"] == 2026 and META["smoke_test"] is False
    for m in cc.MODELS:
        info = META["models"][m]
        assert info["status"] == "OK"
        assert abs(info["test"]["bacc_raw"] - info["test"]["bacc_stored"]) <= 0.0001
        assert abs(info["val"]["bacc_raw"] - info["val"]["bacc_stored"]) <= 0.0006 + 1e-9


def test_raw_probabilities_reproduce_the_stored_bacc(probs):
    yt, yv = np.load(cc.RUN / "tta_labels.npy"), np.load(cc.RUN / "tta_val_labels.npy")
    for m in cc.MODELS:
        stored = json.loads((cc.RUN / f"{m}_meta.json").read_text())
        assert f4(cc.bacc(yt, probs[("test", "raw")][m])) == f4(stored["test_bacc"])
        assert abs(cc.bacc(yv, probs[("val", "raw")][m]) - stored["val_bacc"]) <= 0.0006 + 1e-9


def test_six_model_ensemble_without_tta_matches_canonical(probs):
    yt = np.load(cc.RUN / "tta_labels.npy")
    ens = cc.weighted_ensemble(cc.MODELS, probs[("test", "raw")], cc.weights_from_meta())
    assert f4(cc.bacc(yt, ens)) == "0.8371" == f4(json.loads((cc.RUN / "results.json").read_text())["ensemble"]["test_bacc"])


def test_stored_analysis_equals_recomputation(tmp_path):
    for p in FOLDER.glob("probs_*.npy"):
        (tmp_path / p.name).write_bytes(p.read_bytes())
    recomputed = json.loads(json.dumps(an.analyze(tmp_path, n_boot=10000)))

    def same(a, b, path=""):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                same(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        elif isinstance(a, (bool, str)):
            assert a == b, path
        else:
            assert a == pytest.approx(b, abs=2e-3), path

    same(recomputed, STORED)


def test_readme_per_model_table():
    canon = pd.read_csv(cc.RUN / "ganancia_tta_por_modelo.csv").set_index("Modelo")
    names = {"resnet50": "ResNet-50", "densenet121": "DenseNet-121", "efficientnet_b3": "EfficientNet-B3",
             "vit_b16": "ViT-B/16", "dinov2_b": "DINOv2 ViT-B/14", "biomedclip": "BiomedCLIP"}
    for m in cc.MODELS:
        pm = STORED["per_model"][m]
        line = (f"| {names[m]} | {f4(pm['test_raw'])} | {f4(pm['test_tta'])} | "
                f"{signed(pm['test_tta'] - pm['test_raw'])} | {signed(canon.loc[m, 'Ganancia TTA'])} |")
        assert line in README, line


def test_readme_gap_between_runs():
    canon = pd.read_csv(cc.RUN / "ganancia_tta_por_modelo.csv").set_index("Modelo")["Ganancia TTA"]
    gaps = [STORED["per_model"][m]["test_tta"] - STORED["per_model"][m]["test_raw"] - canon[m] for m in cc.MODELS]
    assert f4(max(abs(g) for g in gaps)) == "0.0165" and "up to 0.0165" in README


def test_readme_contrasts_and_ensembles():
    c, p = STORED["contrasts"], STORED["point_bacc"]
    rows = {
        "Six models against three CNNs, TTA": c["all_tta_minus_cnn_tta"],
        "Three foundation models against three CNNs, TTA": c["foundation_tta_minus_cnn_tta"],
        "DINOv2 with TTA against without": c["dinov2_tta_minus_raw"],
        "Six models with TTA against without": c["all_tta_minus_all_raw"],
    }
    for label, r in rows.items():
        line = (f"| {label} | {signed(r['delta'])} | {signed(r['ci_low'])} to {signed(r['ci_high'])} | "
                f"{f4(r['p_positive'])} |")
        assert line in README, line
    for key in ("cnn_tta", "foundation_tta", "all_tta", "all_raw"):
        assert f4(p[key]) in README
    assert f4(p["all_tta"] - p["cnn_tta"]) == "0.0132" and f4(p["foundation_tta"] - p["cnn_tta"]) == "0.0068"
    canon = pd.read_csv(cc.RUN / "comparativa_ensembles.csv").set_index("Ensemble")["Test BACC"]
    assert f4(canon["Todos"] - canon["CNN (3 originales)"]) == "0.0132"
    assert f4(canon["Fundacionales (3 nuevos)"] - canon["CNN (3 originales)"]) == "0.0087"


def test_readme_subset_search():
    s = STORED["subset_search"]
    assert s["best_subset"] == ["resnet50", "densenet121", "vit_b16", "dinov2_b", "biomedclip"]
    assert (s["best_val_bacc"], s["best_test_bacc"], s["gap"]) == (0.8661, 0.8323, 0.0337)
    assert (s["six_model_rank_on_val"], s["six_model_val_bacc"], s["six_model_test_bacc"]) == (7, 0.859, 0.8415)
    assert s["mean_gap_all_subsets"] == 0.0093 and s["n_subsets"] == 63
    second = s["top5_by_val"][1]
    assert second["models"] == ["resnet50", "vit_b16", "dinov2_b"]
    assert (second["val_bacc"], second["test_bacc"]) == (0.864, 0.8546)
    assert round(second["val_bacc"] - second["test_bacc"], 4) == 0.0094
    assert s["uses_tta"] == {"resnet50": True, "densenet121": False, "efficientnet_b3": True,
                             "vit_b16": True, "dinov2_b": True, "biomedclip": False}
    for x in ("0.8661", "0.8323", "0.0337", "0.0093", "0.0094", "0.8546", "0.8590", "0.8415", "0.8613", "0.8379", "0.0234"):
        assert x in README, x
    assert "(0.8379 and 0.8323 against 0.8415)" in README


def test_readme_normalisation_constants():
    assert META["models"]["vit_b16"]["normalization_native"] == {"mean": [0.5] * 3, "std": [0.5] * 3}
    clip = META["models"]["biomedclip"]["normalization_native"]
    assert [round(v, 3) for v in clip["mean"]] == [0.481, 0.458, 0.408]
    assert [round(v, 3) for v in clip["std"]] == [0.269, 0.261, 0.276]
    assert META["models"]["dinov2_b"]["normalization_native"] == META["models"]["dinov2_b"]["normalization_used"]
    for x in ("0.481, 0.458, 0.408", "0.269, 0.261, 0.276"):
        assert x in README


def test_readme_class_table_matches_counts():
    counts = json.loads((cc.RUN / "class_counts.json").read_text())
    for cls, (tr, va, te) in counts.items():
        assert f"| {tr:,} | {va:,} | {te:,} |" in README, cls
