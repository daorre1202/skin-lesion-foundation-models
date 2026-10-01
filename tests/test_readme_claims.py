import json

from scripts.ce1_common import ROOT, RUN

README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_nevus_share():
    counts = json.loads((RUN / "class_counts.json").read_text())
    total = sum(sum(v) for v in counts.values())
    assert total == 11720 and round(100 * sum(counts["NV"]) / total) == 66
    assert "Nevi are 66% of the images" in README


def test_in_short_numbers():
    short = README.split("## In short")[1].split("##")[0]
    for x in ("+0.0132", "+0.0023 to +0.0244", "39%", "0.804", "0.8415", "0.0234 and 0.0337"):
        assert x in short, x
