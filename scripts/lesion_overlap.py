"""Lesion overlap between the splits, and what it does to the test results.

HAM10000 holds several images of some lesions, and the split of this repository (and of the
thesis repository) is by image. An image is flagged when another image of the same lesion
sits in a split it should not share:
  * validation image: a lesion-mate in train
  * test image: a lesion-mate in train or validation
Only the 10,015 HAM10000 images have a lesion identifier (HAM10000_metadata.csv). The 193
validation and 1,512 test images of ISIC 2018 have none, so for them overlap is unknown.

Two steps:
    python -m scripts.lesion_overlap flags --metadata <HAM10000_metadata.csv> --labels-dir <folder>
        writes results/lesion_overlap/flags.csv (identifiers only, needs the metadata file once)
    python -m scripts.lesion_overlap analyze
        writes results/lesion_overlap/analysis.json from flags.csv and the stored probabilities
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ce1_common import (CLASSES, CNN, FOUNDATION, MODELS, ROOT, RUN, bacc, contrast,  # noqa: E402
                                load_labels, ordered_ids, paired_bootstrap, weighted_ensemble,
                                weights_from_meta)

OUT = ROOT / "results" / "lesion_overlap"
REINFERENCE = ROOT / "results" / "reinference_2026-10"
MALIGNANT = [CLASSES.index(c) for c in ("MEL", "BCC", "AKIEC")]
SUBSETS = ("all", "clean", "overlap", "unknown")


def make_flags(metadata_csv, labels_dir, out_csv):
    meta = pd.read_csv(metadata_csv)
    lesion = dict(zip(meta["image_id"], meta["lesion_id"]))
    assignment = json.loads((RUN / "split_assignment.json").read_text())
    labels = load_labels(labels_dir)
    by_lesion = {}
    for img, split in assignment.items():
        if img in lesion:
            by_lesion.setdefault(lesion[img], []).append((img, split))
    rows = []
    for split, shared in (("val", {"train"}), ("test", {"train", "val"})):
        for pos, img in enumerate(ordered_ids(labels, assignment, split)):
            known = img in lesion
            overlap = known and any(s in shared and j != img for j, s in by_lesion[lesion[img]])
            rows.append((img, split, pos, int(known), int(overlap)))
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["image_id", "split", "position", "has_lesion_id", "overlap"])
        w.writerows(rows)
    return rows


def subset_masks(flags, split="test"):
    f = flags[flags["split"] == split].sort_values("position")
    known, over = f["has_lesion_id"].to_numpy() == 1, f["overlap"].to_numpy() == 1
    return {"all": np.ones(len(f), bool), "clean": known & ~over, "overlap": over, "unknown": ~known}


def rec(y, pred, cls):
    return float((pred[y == cls] == cls).mean())


def point(y, pred):
    return {"n": int(len(y)), "bacc": float(np.mean([rec(y, pred, c) for c in range(len(CLASSES)) if (y == c).any()])),
            "malignant_bacc": float(np.mean([rec(y, pred, c) for c in MALIGNANT])),
            "class_counts": np.bincount(y, minlength=len(CLASSES)).tolist()}


def ci(values):
    lo, hi = np.percentile(values, [2.5, 97.5])
    return [float(lo), float(hi)]


def analyze(flags_csv=OUT / "flags.csv", out_json=OUT / "analysis.json", n_boot=10000, seed=42):
    flags = pd.read_csv(flags_csv)
    yt = np.load(RUN / "tta_labels.npy")
    masks = subset_masks(flags)
    val_masks = subset_masks(flags, "val")
    w = weights_from_meta()
    re_probs = {m: np.load(REINFERENCE / f"probs_{m}_test_tta.npy") for m in MODELS}
    systems = {
        "canonical_argmax": np.load(RUN / "tta_sum_probs.npy").argmax(1),
        "canonical_thresholds": np.load(RUN / "tta_thresh_preds.npy"),
        "reinference_six": weighted_ensemble(MODELS, re_probs, w).argmax(1),
        "reinference_cnn": weighted_ensemble(CNN, re_probs, w).argmax(1),
        "reinference_foundation": weighted_ensemble(FOUNDATION, re_probs, w).argmax(1),
    }
    result = {"n_boot": n_boot, "seed": seed,
              "counts": {"test": {k: int(v.sum()) for k, v in masks.items()},
                         "val": {k: int(v.sum()) for k, v in val_masks.items()}},
              "systems": {}, "clean_subset_intervals": {}, "threshold_gain": {}, "contrasts": {}}
    for name, pred in systems.items():
        result["systems"][name] = {s: point(yt[m], pred[m]) for s, m in masks.items()}

    for name in ("canonical_argmax", "canonical_thresholds", "reinference_six"):
        m = masks["clean"]
        boot = paired_bootstrap(yt[m], {"x": systems[name][m]}, n_boot, seed)["x"]
        result["clean_subset_intervals"][name] = {"bacc": ci(boot)}

    for s in ("all", "clean", "overlap"):
        m = masks[s]
        boot = paired_bootstrap(yt[m], {"a": systems["canonical_argmax"][m], "t": systems["canonical_thresholds"][m]},
                                n_boot, seed, classes=MALIGNANT)
        d = boot["t"] - boot["a"]
        result["threshold_gain"][s] = {"delta": float(np.mean(
            [rec(yt[m], systems["canonical_thresholds"][m], c) for c in MALIGNANT])
            - np.mean([rec(yt[m], systems["canonical_argmax"][m], c) for c in MALIGNANT])),
            "ci": ci(d), "p_positive": float((d > 0).mean())}

    for s in ("all", "clean", "overlap"):
        m = masks[s]
        preds = {k: systems[k][m] for k in ("reinference_six", "reinference_cnn", "reinference_foundation")}
        boot = paired_bootstrap(yt[m], preds, n_boot, seed)
        pt = {k: result["systems"][k][s]["bacc"] for k in preds}
        result["contrasts"][s] = {
            "six_minus_cnn": contrast(boot, "reinference_six", "reinference_cnn", pt["reinference_six"], pt["reinference_cnn"]),
            "foundation_minus_cnn": contrast(boot, "reinference_foundation", "reinference_cnn",
                                             pt["reinference_foundation"], pt["reinference_cnn"])}
    Path(out_json).write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("flags")
    f.add_argument("--metadata", required=True)
    f.add_argument("--labels-dir", required=True)
    sub.add_parser("analyze")
    args = ap.parse_args()
    if args.cmd == "flags":
        rows = make_flags(args.metadata, args.labels_dir, OUT / "flags.csv")
        t = [r for r in rows if r[1] == "test"]
        print(f"test: {sum(r[3] for r in t)} images with a lesion id, {sum(r[4] for r in t)} overlap")
    else:
        r = analyze()
        print(json.dumps(r["counts"], indent=1))


if __name__ == "__main__":
    main()
