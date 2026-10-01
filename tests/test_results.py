import hashlib
import json
from collections import Counter

import numpy as np
import pandas as pd
import pytest

from scripts import verify_results as v

RUN = v.RUN
ROOT = v.ROOT
README = (ROOT / "README.md").read_text(encoding="utf-8")
MODELS = ["resnet50", "densenet121", "efficientnet_b3", "vit_b16", "dinov2_b", "biomedclip"]


@pytest.fixture(scope="module")
def rec():
    return v.recompute()


@pytest.fixture(scope="module")
def results():
    return json.loads((RUN / "results.json").read_text())


def fmt(x, nd=4):
    return f"{x:.{nd}f}"


def test_split_counts():
    split = json.loads((RUN / "split_assignment.json").read_text())
    counts = Counter(split.values())
    assert (counts["train"], counts["val"], counts["test"]) == (7030, 2342, 2348)
    assert len(split) == 11720


def test_class_counts_match_split():
    cc = json.loads((RUN / "class_counts.json").read_text())
    assert [sum(cc[c][i] for c in cc) for i in range(3)] == [7030, 2342, 2348]
    assert json.loads((RUN / "classes_used.json").read_text()) == v.CLASSES


def test_array_shapes(rec):
    a = rec["arrays"]
    assert a["test_probs"].shape == (2348, 7) and a["val_probs"].shape == (2342, 7)
    assert np.allclose(a["test_probs"].sum(1), 1, atol=1e-4)
    assert np.allclose(a["val_probs"].sum(1), 1, atol=1e-4)


def test_ensemble_bacc_from_arrays(rec, results):
    assert fmt(rec["bacc_argmax"]) == fmt(results["tta_ensemble"]["test_bacc"]) == "0.8415"


def test_thresholds_recalibrated_from_validation(rec):
    saved = json.loads((RUN / "calibrated_thresholds.json").read_text())
    for name in v.THRESHOLD_PRIORITY:
        assert rec["thresholds"][name] == pytest.approx(saved[name], abs=1e-9)
    assert fmt(rec["thresholds"]["MEL"], 3) == "0.307"
    assert fmt(rec["thresholds"]["AKIEC"], 3) == "0.386"


def test_threshold_predictions_match_saved_array(rec):
    assert (rec["thresh_preds"] == rec["arrays"]["saved_thresh_preds"]).all()


def test_clinical_numbers(rec, results):
    assert fmt(rec["bacc_thresholds"]) == fmt(results["clinical"]["test_bacc"]) == "0.8499"
    assert fmt(rec["malignant_thresholds"]) == fmt(results["clinical"]["bacc_malignant"]) == "0.8250"
    assert fmt(rec["malignant_argmax"]) == "0.7861"
    assert fmt(rec["malignant_thresholds"] - rec["malignant_argmax"]) == "0.0388"
    assert fmt(rec["bacc_thresholds"] - rec["bacc_argmax"]) == "0.0084"


def test_per_class_table_matches_stored_csv(rec):
    stored = pd.read_csv(RUN / "metricas_clinicas_umbrales_clinicos.csv")
    pd.testing.assert_frame_equal(
        stored.reset_index(drop=True), rec["metrics_thresholds"].reset_index(drop=True),
        check_dtype=False)


def test_vit_table_is_vit_test_bacc():
    t = pd.read_csv(RUN / "metricas_clinicas_vit_b16.csv")
    meta = json.loads((RUN / "vit_b16_meta.json").read_text())
    assert fmt(t["Sensibilidad"].mean()) == fmt(meta["test_bacc"]) == "0.8009"


@pytest.mark.parametrize("csv", ["metricas_clinicas_umbrales_clinicos.csv", "metricas_clinicas_vit_b16.csv"])
def test_confusion_tables_consistent(csv):
    t = pd.read_csv(RUN / csv)
    assert ((t["TP"] + t["FN"]).tolist() == [261, 1548, 125, 77, 269, 32, 36])
    assert np.allclose(t["Sensibilidad"], (t["TP"] / (t["TP"] + t["FN"])).round(4))
    assert np.allclose(t["Especificidad"], (t["TN"] / (t["TN"] + t["FP"])).round(4))


def test_per_class_readme_rows(rec):
    for _, row in rec["metrics_argmax"].iterrows():
        t = rec["metrics_thresholds"].set_index("Clase").loc[row["Clase"]]
        line = (f"| {row['Clase']} | {row['Sensibilidad']:.4f} | "
                f"{t['Sensibilidad']:.4f} | {t['Especificidad']:.4f} |")
        assert line in README, line


def test_individual_models_table(results):
    gain = pd.read_csv(RUN / "ganancia_tta_por_modelo.csv").set_index("Modelo")
    for m in MODELS:
        meta = json.loads((RUN / f"{m}_meta.json").read_text())
        j = results["models"][m]
        assert round(meta["val_bacc"], 4) == j["val_bacc"]
        assert round(meta["test_bacc"], 4) == j["test_bacc"]
        for x in (j["val_bacc"], j["test_bacc"], gain.loc[m, "Test BACC + TTA"]):
            assert fmt(x) in README, (m, x)
        g = gain.loc[m, "Ganancia TTA"]
        assert f"{g:+.4f}".replace("-", "-") in README, (m, g)


def test_ensembles_table(results):
    comp = pd.read_csv(RUN / "comparativa_ensembles.csv").set_index("Ensemble")["Test BACC"]
    assert comp["CNN (3 originales)"] == 0.8283
    assert comp["Fundacionales (3 nuevos)"] == 0.8370
    assert comp["Todos"] == 0.8415 == round(results["tta_ensemble"]["test_bacc"], 4)
    assert round(results["ensemble"]["test_bacc"], 4) == 0.8371
    assert fmt(comp["Todos"] - comp["CNN (3 originales)"]) == "0.0132"
    assert fmt(comp["Fundacionales (3 nuevos)"] - comp["CNN (3 originales)"]) == "0.0087"
    for x in ("0.8283", "0.8370", "0.8415", "0.8371", "0.8499", "+0.0132", "+0.0087"):
        assert x in README


def test_subset_search_and_selection_gap():
    s = pd.read_csv(RUN / "busqueda_subconjuntos.csv")
    assert len(s) == 63
    s = s.sort_values("Val_BACC", ascending=False).reset_index(drop=True)
    assert s.loc[0, "Modelos"] == "resnet50, vit_b16, dinov2_b" and s.loc[0, "Val_BACC"] == 0.8613
    comp = pd.read_csv(RUN / "comparativa_ensembles.csv").set_index("Ensemble")["Test BACC"]
    best_test = comp["Mejor subconjunto (val)"]
    assert best_test == 0.8379
    assert fmt(s.loc[0, "Val_BACC"] - best_test) == "0.0234"
    six = s[s["N"] == 6].index[0] + 1
    assert six == 5 and s[s["N"] == 6]["Val_BACC"].iloc[0] == 0.8552
    for x in ("0.8613", "0.8379", "0.0234", "0.8552", "fifth of 63"):
        assert x in README


def test_tta_decision_csv():
    d = pd.read_csv(RUN / "tta_decision_por_modelo.csv").set_index("Modelo")
    assert d["Usa_TTA"].to_dict() == {m: m == "dinov2_b" for m in MODELS}


def test_total_training_minutes():
    secs = sum(json.loads((RUN / f"{m}_meta.json").read_text())["tiempo_segundos"] for m in MODELS)
    assert int(secs // 60) == 618 and "618 minutes" in README


def test_gamma_provenance_numbers():
    text = (ROOT / "results" / "earlier_session_2026-07-31" / "stdout_from_notebook.txt").read_text(encoding="utf-8")
    assert "Val BACC: 0.8185" in text and "Test BACC: 0.7955" in text      # ViT-B/16, gamma 0
    assert "Val BACC: 0.7734" in text and "Test BACC: 0.7439" in text      # BiomedCLIP, gamma 0
    for m, line in (("vit_b16", "0.8259"), ("biomedclip", "0.7949")):
        assert fmt(json.loads((RUN / f"{m}_meta.json").read_text())["val_bacc"]) == line
    assert round(0.8009 - 0.7955, 4) == 0.0054 and round(0.7882 - 0.7439, 4) == 0.0443
    for x in ("0.8185", "0.8259", "0.7734", "0.7949", "0.7955", "0.8009", "0.7439", "0.7882", "+0.0054", "+0.0443", "-0.0141"):
        assert x in README


def test_earlier_session_matches_canonical_for_four_models():
    text = (ROOT / "results" / "earlier_session_2026-07-31" / "stdout_from_notebook.txt").read_text(encoding="utf-8")
    for m, (val, test, epoch, t) in {
        "resnet50": ("0.7977", "0.7881", 34, "58m 4s"),
        "densenet121": ("0.7952", "0.7862", 30, "51m 0s"),
        "efficientnet_b3": ("0.7975", "0.7940", 11, "57m 57s"),
        "dinov2_b": ("0.8013", "0.7950", 25, "193m 17s"),
    }.items():
        meta = json.loads((RUN / f"{m}_meta.json").read_text())
        assert fmt(meta["val_bacc"]) == val and fmt(meta["test_bacc"]) == test and meta["best_epoch"] == epoch
        assert f"epoch {epoch} | Val BACC: {val}" in text and f"Tiempo: {t}" in text


def test_log_says_resume():
    assert "REANUDAR" in (RUN / "ejecucion_log.txt").read_text(encoding="utf-8")


def test_manifest_hashes():
    listed = {}
    for line in (RUN / "SHA256SUMS.txt").read_text().splitlines():
        h, name = line.split("  ", 1)
        listed[name] = h
    actual = {p.name for p in RUN.iterdir()} - {"GLOSSARY.md", "SHA256SUMS.txt"}
    assert set(listed) == actual
    for name, h in listed.items():
        assert hashlib.sha256((RUN / name).read_bytes()).hexdigest() == h, name


def test_one_vs_rest_sensitivity_at_thresholds(rec):
    a = rec["arrays"]
    got = {}
    for name in v.THRESHOLD_PRIORITY:
        idx, th = v.CLASSES.index(name), rec["thresholds"][name]
        for split, (p, y) in {"val": (a["val_probs"], a["val_labels"]),
                              "test": (a["test_probs"], a["test_labels"])}.items():
            pred, lab = p[:, idx] >= th, y == idx
            got[(name, split)] = (pred & lab).sum() / lab.sum()
    assert fmt(got[("MEL", "val")]) == "0.8506" and fmt(got[("AKIEC", "val")]) == "0.7733"
    assert fmt(got[("MEL", "test")]) == "0.8238" and fmt(got[("AKIEC", "test")]) == "0.7143"
    assert got[("MEL", "val")] >= 0.85 and got[("AKIEC", "val")] >= 0.75
    assert got[("MEL", "test")] < 0.85 and got[("AKIEC", "test")] < 0.75
    for x in ("0.8506", "0.7733", "0.8238", "0.7143", "0.8276", "0.7273"):
        assert x in README
    assert (a["val_labels"] == 3).sum() == 75 and (a["test_labels"] == 3).sum() == 77
