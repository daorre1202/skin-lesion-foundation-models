"""Shared helpers for the re-inference and its analysis. No torch needed here."""
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import balanced_accuracy_score

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / "results" / "canonical_run_2026-08-04"

CLASSES = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
MODELS = ["resnet50", "densenet121", "efficientnet_b3", "vit_b16", "dinov2_b", "biomedclip"]
CNN = ["resnet50", "densenet121", "efficientnet_b3"]
FOUNDATION = ["vit_b16", "dinov2_b", "biomedclip"]
IMG_SIZE = {"resnet50": 224, "densenet121": 224, "efficientnet_b3": 300,
            "vit_b16": 224, "dinov2_b": 224, "biomedclip": 224}
GT_FILES = {"train": "ISIC2018_Task3_Training_GroundTruth.csv",
            "val": "ISIC2018_Task3_Validation_GroundTruth.csv",
            "test": "ISIC2018_Task3_Test_GroundTruth.csv"}


# ---------------------------------------------------------------- data order
def load_labels(labels_dir):
    """image id -> class name, from the three official ground-truth CSV files."""
    frames = [pd.read_csv(Path(labels_dir) / name) for name in GT_FILES.values()]
    df = pd.concat(frames, ignore_index=True)
    onehot = df[CLASSES].to_numpy()
    return dict(zip(df["image"], [CLASSES[i] for i in onehot.argmax(axis=1)]))


def ordered_ids(labels, assignment, split):
    """Image ids of one split in the order the training notebook evaluated them:
    classes in CLASSES order, file names sorted inside each class."""
    ids = []
    for cls in CLASSES:
        ids += sorted(i for i, s in assignment.items() if s == split and labels[i] == cls)
    return ids


def label_array(labels, ids):
    return np.array([CLASSES.index(labels[i]) for i in ids])


def index_images(roots):
    """image id -> path, searching every root recursively. First root wins."""
    found = {}
    for root in roots:
        for path in sorted(Path(root).expanduser().rglob("*")):
            if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                found.setdefault(path.stem, path)
    return found


# ------------------------------------------------------------------ metrics
def weights_from_meta(run=RUN):
    """Ensemble weights: each model's best validation BACC, as in the canonical run."""
    return {m: json.loads((Path(run) / f"{m}_meta.json").read_text())["val_bacc"] for m in MODELS}


def weighted_ensemble(names, probs, weights):
    total = sum(weights[n] for n in names)
    out = None
    for n in names:
        part = probs[n] * (weights[n] / total)
        out = part if out is None else out + part
    return out


def bacc(y, probs):
    return float(balanced_accuracy_score(y, probs.argmax(axis=1)))


def paired_bootstrap(y, preds, n_boot=10000, seed=42, chunk=500, classes=None):
    """BACC of several prediction vectors on the same bootstrap resamples.

    preds: dict name -> predicted class indices. Returns dict name -> array (n_boot,).
    All predictions share the resampled indices, so differences are paired.
    classes: restrict the mean recall to these class indices (default: all seven)."""
    rng = np.random.default_rng(seed)
    n, k = len(y), len(CLASSES)
    use = np.zeros(k, dtype=bool)
    use[list(range(k)) if classes is None else list(classes)] = True
    onehot = y[:, None] == np.arange(k)[None, :]
    correct = {name: (p[:, None] == np.arange(k)[None, :]) & onehot for name, p in preds.items()}
    out = {name: np.empty(n_boot) for name in preds}
    for start in range(0, n_boot, chunk):
        b = min(chunk, n_boot - start)
        idx = rng.integers(0, n, size=(b, n))
        support = onehot[idx].sum(axis=1)
        for name, c in correct.items():
            recall = np.where((support > 0) & use[None, :], c[idx].sum(axis=1) / np.maximum(support, 1), np.nan)
            out[name][start:start + b] = np.nanmean(recall, axis=1)
    return out


def contrast(boot, plus, minus, point_plus, point_minus):
    """Point difference, 95% percentile interval and P(delta > 0) of two BACC series."""
    delta = boot[plus] - boot[minus]
    lo, hi = np.percentile(delta, [2.5, 97.5])
    return {"delta": float(point_plus - point_minus), "ci_low": float(lo), "ci_high": float(hi),
            "p_positive": float((delta > 0).mean())}


# ------------------------------------------------------------ subset search
def subset_search(val_probs, test_probs, weights, yv, yt):
    """Best subset on validation among all 63, confirmed once on test.

    Also returns the validation-to-test gap averaged over all subsets, which is the
    optimism every subset carries because checkpoints and weights were chosen on validation,
    and the five subsets that rank highest on validation."""
    rows = []
    best = None
    for r in range(1, len(MODELS) + 1):
        for combo in itertools.combinations(MODELS, r):
            vb = bacc(yv, weighted_ensemble(list(combo), val_probs, weights))
            tb = bacc(yt, weighted_ensemble(list(combo), test_probs, weights))
            rows.append({"models": list(combo), "n": r, "val_bacc": round(vb, 4),
                         "test_bacc": round(tb, 4)})
            if best is None or vb > best[0]:
                best = (vb, tb, list(combo))
    table = sorted(rows, key=lambda x: -x["val_bacc"])
    six = next(i for i, row in enumerate(table) if row["n"] == len(MODELS)) + 1
    six_row = next(r for r in table if r["n"] == len(MODELS))
    return {"best_subset": best[2], "best_val_bacc": round(best[0], 4),
            "best_test_bacc": round(best[1], 4), "gap": round(best[0] - best[1], 4),
            "six_model_rank_on_val": six, "six_model_val_bacc": six_row["val_bacc"],
            "six_model_test_bacc": six_row["test_bacc"], "n_subsets": len(rows),
            "mean_gap_all_subsets": round(float(np.mean([r["val_bacc"] - r["test_bacc"] for r in rows])), 4),
            "top5_by_val": table[:5]}
