import json

import numpy as np
import pandas as pd
import pytest

from scripts import ce1_common as cc
from scripts import lesion_overlap as lo

FLAGS = lo.OUT / "flags.csv"
STORED = json.loads((lo.OUT / "analysis.json").read_text())
README = (cc.ROOT / "README.md").read_text(encoding="utf-8")


def f4(x):
    return f"{x:.4f}"


def sg(x):
    return f"{x:+.4f}"


@pytest.fixture(scope="module")
def flags():
    return pd.read_csv(FLAGS)


def test_flags_cover_every_validation_and_test_image(flags):
    assert (flags["split"] == "val").sum() == 2342 and (flags["split"] == "test").sum() == 2348
    for split in ("val", "test"):
        f = flags[flags["split"] == split]
        assert f["position"].tolist() == list(range(len(f)))
        assert not ((f["has_lesion_id"] == 0) & (f["overlap"] == 1)).any()
    split = json.loads((cc.RUN / "split_assignment.json").read_text())
    assert all(split[i] == s for i, s in zip(flags["image_id"], flags["split"]))


def test_counts_in_the_readme():
    c = STORED["counts"]
    assert c["test"] == {"all": 2348, "clean": 1212, "overlap": 779, "unknown": 357}
    assert c["val"] == {"all": 2342, "clean": 1423, "overlap": 577, "unknown": 342}
    assert round(100 * 779 / (1212 + 779)) == 39 and round(100 * 577 / (1423 + 577)) == 29
    for x in ("779 (39%)", "577 of the 2,000", "(29%)", "1,212", "357", "| 2,348 |", "| 779 |"):
        assert x in README, x


def test_stored_analysis_equals_recomputation(tmp_path):
    out = tmp_path / "analysis.json"
    recomputed = json.loads(json.dumps(lo.analyze(FLAGS, out, n_boot=10000)))

    def same(a, b, path=""):
        if isinstance(a, dict):
            assert a.keys() == b.keys(), path
            for k in a:
                same(a[k], b[k], f"{path}/{k}")
        elif isinstance(a, list):
            assert len(a) == len(b), path
            for i, (x, y) in enumerate(zip(a, b)):
                same(x, y, f"{path}[{i}]")
        else:
            assert a == pytest.approx(b, abs=2e-3), path

    same(recomputed, STORED)


def test_full_test_numbers_match_the_canonical_run():
    s = STORED["systems"]
    assert f4(s["canonical_argmax"]["all"]["bacc"]) == "0.8415"
    assert f4(s["canonical_thresholds"]["all"]["bacc"]) == "0.8499"
    assert f4(s["canonical_argmax"]["all"]["malignant_bacc"]) == "0.7861"
    assert f4(s["canonical_thresholds"]["all"]["malignant_bacc"]) == "0.8250"
    assert f4(s["reinference_six"]["all"]["bacc"]) == "0.8427"


def test_readme_group_table():
    s = STORED["systems"]
    labels = {"all": "All test images", "clean": "No lesion-mate in train or validation",
              "overlap": "Lesion-mate in train or validation", "unknown": "No lesion identifier (official ISIC 2018 sets)"}
    for g, label in labels.items():
        a, t = s["canonical_argmax"][g], s["canonical_thresholds"][g]
        line = (f"| {label} | {a['n']:,} | {f4(a['bacc'])} | {f4(t['bacc'])} | "
                f"{f4(a['malignant_bacc'])} to {f4(t['malignant_bacc'])} |")
        assert line in README, line
    assert s["canonical_argmax"]["clean"]["class_counts"] == [62, 943, 41, 35, 111, 9, 11]
    assert "62 MEL, 943 NV, 41 BCC, 35 AKIEC, 111 BKL, 9 DF and 11 VASC" in README


def test_readme_intervals_and_gains():
    ci = STORED["clean_subset_intervals"]
    lo_a, hi_a = ci["canonical_argmax"]["bacc"]
    lo_t, hi_t = ci["canonical_thresholds"]["bacc"]
    assert (f"{lo_a:.3f}", f"{hi_a:.3f}") == ("0.744", "0.862") and (f"{lo_t:.3f}", f"{hi_t:.3f}") == ("0.750", "0.866")
    assert lo_a < STORED["systems"]["canonical_argmax"]["all"]["bacc"] < hi_a
    for x in ("0.744 to 0.862", "0.750 to 0.866"):
        assert x in README
    g = STORED["threshold_gain"]
    for key, label in (("all", "on all images"), ("clean", "without lesion-mate"), ("overlap", "with one")):
        d, (a, b) = g[key]["delta"], g[key]["ci"]
        assert f"{sg(d)} {label} ({sg(a)} to {sg(b)})" in README or f"{sg(d)} ({sg(a)} to {sg(b)})" in README \
            or f"{sg(d)} {label} ({sg(a)} to {sg(b)})" in README, (key, sg(d), sg(a), sg(b))
    assert g["clean"]["ci"][0] > 0


def test_readme_contrasts_on_clean_images():
    c = STORED["contrasts"]
    six, fm = c["clean"]["six_minus_cnn"], c["clean"]["foundation_minus_cnn"]
    assert f"{sg(six['delta'])} ({sg(six['ci_low'])} to {sg(six['ci_high'])})" in README
    assert f"{sg(fm['delta'])} ({sg(fm['ci_low'])} to {sg(fm['ci_high'])})" in README
    assert six["ci_low"] < 0 < six["ci_high"] and fm["ci_low"] < 0 < fm["ci_high"]
    full = c["all"]["six_minus_cnn"]
    assert f4(full["delta"]) == "0.0132" and full["ci_low"] > 0


def test_flags_agree_with_unknown_group_being_the_non_ham_images(flags):
    t = flags[flags["split"] == "test"]
    assert (t["has_lesion_id"] == 0).sum() == 357
