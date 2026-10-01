"""Recompute the headline numbers of the canonical run from the saved arrays.

Everything here runs on CPU in a few seconds. It uses only files stored under
results/canonical_run_2026-08-04: the summed TTA probabilities of the six-model
ensemble on validation and test, the labels, and the per-class confusion tables.

The calibration and metric functions are a port of the notebook cells
"Threshold calibration" (calibrate_threshold, apply_clinical_thresholds,
compute_per_class_metrics) without logging.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "canonical_run_2026-08-04"

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MALIGNANT = ["MEL", "BCC", "AKIEC"]
THRESHOLD_PRIORITY = ["MEL", "AKIEC"]
CLINICAL_THRESHOLDS = {
    "MEL": {"sensitivity_target": 0.85, "specificity_floor": 0.85},
    "AKIEC": {"sensitivity_target": 0.75, "specificity_floor": 0.70},
}


def load_arrays(run=RUN):
    return {
        "test_probs": np.load(run / "tta_sum_probs.npy"),
        "test_labels": np.load(run / "tta_labels.npy"),
        "val_probs": np.load(run / "tta_val_sum.npy"),
        "val_labels": np.load(run / "tta_val_labels.npy"),
        "saved_thresh_preds": np.load(run / "tta_thresh_preds.npy"),
    }


def calibrate_threshold(val_probs, val_labels, cls_idx, sensitivity_target, specificity_floor):
    """Highest theta in a 199-point grid with sens >= target and spec >= floor."""

    def search(floor):
        candidates = []
        for theta in np.linspace(0.01, 0.99, 199):
            preds = (val_probs[:, cls_idx] >= theta).astype(int)
            lbin = (val_labels == cls_idx).astype(int)
            tp = ((preds == 1) & (lbin == 1)).sum()
            fp = ((preds == 1) & (lbin == 0)).sum()
            fn = ((preds == 0) & (lbin == 1)).sum()
            tn = ((preds == 0) & (lbin == 0)).sum()
            sens = tp / max(tp + fn, 1)
            spec = tn / max(tn + fp, 1)
            if spec < floor or sens < sensitivity_target:
                continue
            candidates.append((theta, sens, spec))
        return candidates

    candidates = search(specificity_floor) or search(0.50)
    if not candidates:
        return 0.5
    return max(candidates, key=lambda c: c[0])[0]


def apply_clinical_thresholds(probs, thresholds):
    preds = probs.argmax(axis=1).copy()
    for name in reversed(THRESHOLD_PRIORITY):
        if name in thresholds:
            preds[probs[:, CLASSES.index(name)] > thresholds[name]] = CLASSES.index(name)
    return preds


def per_class_metrics(preds, labels):
    cm = confusion_matrix(labels, preds, labels=list(range(len(CLASSES))))
    tp = np.diag(cm)
    fp = cm.sum(0) - tp
    fn = cm.sum(1) - tp
    tn = cm.sum() - (tp + fp + fn)
    return pd.DataFrame({
        "Clase": CLASSES, "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "Sensibilidad": np.round(tp / np.maximum(tp + fn, 1), 4),
        "Especificidad": np.round(tn / np.maximum(tn + fp, 1), 4),
    })


def malignant_bacc(metrics):
    """Mean one-vs-rest sensitivity over MEL, BCC and AKIEC."""
    return float(metrics[metrics["Clase"].isin(MALIGNANT)]["Sensibilidad"].mean())


def recompute(run=RUN):
    a = load_arrays(run)
    thresholds = {
        name: float(calibrate_threshold(
            a["val_probs"], a["val_labels"], CLASSES.index(name),
            t["sensitivity_target"], t["specificity_floor"]))
        for name, t in CLINICAL_THRESHOLDS.items()
    }
    argmax_preds = a["test_probs"].argmax(axis=1)
    thresh_preds = apply_clinical_thresholds(a["test_probs"], thresholds)
    m_argmax = per_class_metrics(argmax_preds, a["test_labels"])
    m_thresh = per_class_metrics(thresh_preds, a["test_labels"])
    return {
        "thresholds": thresholds,
        "thresh_preds": thresh_preds,
        "bacc_argmax": balanced_accuracy_score(a["test_labels"], argmax_preds),
        "bacc_thresholds": balanced_accuracy_score(a["test_labels"], thresh_preds),
        "val_bacc_argmax": balanced_accuracy_score(a["val_labels"], a["val_probs"].argmax(axis=1)),
        "malignant_argmax": malignant_bacc(m_argmax),
        "malignant_thresholds": malignant_bacc(m_thresh),
        "metrics_argmax": m_argmax,
        "metrics_thresholds": m_thresh,
        "arrays": a,
    }


def main():
    r = recompute()
    saved = json.loads((RUN / "calibrated_thresholds.json").read_text())
    results = json.loads((RUN / "results.json").read_text())
    ok = True

    def check(label, got, expected, tol=1e-4):
        nonlocal ok
        good = abs(got - expected) <= tol
        ok &= good
        print(f"{'OK  ' if good else 'FAIL'} {label}: recomputed {got:.4f} | stored {expected:.4f}")

    check("TTA ensemble BACC (argmax)", r["bacc_argmax"], results["tta_ensemble"]["test_bacc"])
    for name in THRESHOLD_PRIORITY:
        check(f"theta {name}", r["thresholds"][name], saved[name], tol=1e-9)
    check("TTA ensemble + thresholds BACC", r["bacc_thresholds"], results["clinical"]["test_bacc"])
    check("malignant BACC with thresholds", r["malignant_thresholds"], results["clinical"]["bacc_malignant"])
    same = bool((r["thresh_preds"] == r["arrays"]["saved_thresh_preds"]).all())
    ok &= same
    print(f"{'OK  ' if same else 'FAIL'} threshold predictions identical to the saved array")
    print(f"INFO malignant BACC with argmax: {r['malignant_argmax']:.4f}")
    print(f"INFO validation BACC (argmax, TTA ensemble): {r['val_bacc_argmax']:.4f}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
