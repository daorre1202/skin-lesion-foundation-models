"""Analysis of the per-model re-inference: ensembles, paired bootstrap, subset search.

Usage:
    python scripts/analyze_reinference.py [--dir results/reinference_2026-10]

Reads probs_<model>_<split>_<raw|tta>.npy written by reinfer_per_model.py and the
canonical labels. Writes analysis.json next to the arrays. CPU only.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.ce1_common import (CLASSES, CNN, FOUNDATION, MODELS, ROOT, RUN, bacc, contrast,
                                paired_bootstrap, subset_search, weighted_ensemble,
                                weights_from_meta)

DEFAULT_DIR = ROOT / "results" / "reinference_2026-10"


def load_probs(folder):
    probs = {}
    for split in ("val", "test"):
        for mode in ("raw", "tta"):
            probs[(split, mode)] = {m: np.load(Path(folder) / f"probs_{m}_{split}_{mode}.npy")
                                    for m in MODELS}
    return probs


def analyze(folder, run=RUN, n_boot=10000, seed=42):
    probs = load_probs(folder)
    yt, yv = np.load(run / "tta_labels.npy"), np.load(run / "tta_val_labels.npy")
    w = weights_from_meta(run)
    test_raw, test_tta = probs[("test", "raw")], probs[("test", "tta")]
    val_raw, val_tta = probs[("val", "raw")], probs[("val", "tta")]

    per_model = {m: {"val_raw": bacc(yv, val_raw[m]), "val_tta": bacc(yv, val_tta[m]),
                     "test_raw": bacc(yt, test_raw[m]), "test_tta": bacc(yt, test_tta[m])}
                 for m in MODELS}

    ens = {
        "cnn_tta": weighted_ensemble(CNN, test_tta, w),
        "foundation_tta": weighted_ensemble(FOUNDATION, test_tta, w),
        "all_tta": weighted_ensemble(MODELS, test_tta, w),
        "all_raw": weighted_ensemble(MODELS, test_raw, w),
    }
    point = {k: bacc(yt, v) for k, v in ens.items()}
    point.update({f"{m}_raw": per_model[m]["test_raw"] for m in MODELS})
    point.update({f"{m}_tta": per_model[m]["test_tta"] for m in MODELS})

    preds = {k: v.argmax(axis=1) for k, v in ens.items()}
    for m in MODELS:
        preds[f"{m}_raw"] = test_raw[m].argmax(axis=1)
        preds[f"{m}_tta"] = test_tta[m].argmax(axis=1)
    boot = paired_bootstrap(yt, preds, n_boot=n_boot, seed=seed)

    contrasts = {
        "all_tta_minus_cnn_tta": contrast(boot, "all_tta", "cnn_tta", point["all_tta"], point["cnn_tta"]),
        "foundation_tta_minus_cnn_tta": contrast(boot, "foundation_tta", "cnn_tta",
                                                 point["foundation_tta"], point["cnn_tta"]),
        "dinov2_tta_minus_raw": contrast(boot, "dinov2_b_tta", "dinov2_b_raw",
                                         point["dinov2_b_tta"], point["dinov2_b_raw"]),
        "all_tta_minus_all_raw": contrast(boot, "all_tta", "all_raw", point["all_tta"], point["all_raw"]),
    }

    use_tta = {m: per_model[m]["val_tta"] > per_model[m]["val_raw"] for m in MODELS}
    val_final = {m: (val_tta if use_tta[m] else val_raw)[m] for m in MODELS}
    test_final = {m: (test_tta if use_tta[m] else test_raw)[m] for m in MODELS}
    search = subset_search(val_final, test_final, w, yv, yt)
    search["uses_tta"] = use_tta

    result = {"n_boot": n_boot, "seed": seed, "per_model": per_model,
              "point_bacc": point, "contrasts": contrasts, "subset_search": search}
    (Path(folder) / "analysis.json").write_text(json.dumps(result, indent=2))
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default=str(DEFAULT_DIR))
    ap.add_argument("--n-boot", type=int, default=10000)
    args = ap.parse_args()
    r = analyze(args.dir, n_boot=args.n_boot)
    print("point BACC:", {k: round(v, 4) for k, v in r["point_bacc"].items() if k.endswith("tta") or k == "all_raw"})
    for name, c in r["contrasts"].items():
        print(f"{name}: {c['delta']:+.4f}  95% CI [{c['ci_low']:+.4f}, {c['ci_high']:+.4f}]  P(>0)={c['p_positive']:.3f}")
    s = r["subset_search"]
    print(f"best subset on validation: {s['best_subset']}  val {s['best_val_bacc']}  test {s['best_test_bacc']}  gap {s['gap']}")
    print(f"six-model ensemble rank on validation: {s['six_model_rank_on_val']} of {s['n_subsets']}")


if __name__ == "__main__":
    main()
