from scripts.ce1_common import ROOT

FOLDER = ROOT / "results" / "dataloader_rng_check"
README = (ROOT / "README.md").read_text(encoding="utf-8")


def test_stored_check_output():
    lines = (FOLDER / "output.txt").read_text().splitlines()
    assert lines[0] == "albumentations 2.0.8"
    assert lines[1] == "num_workers=0: dos pasadas identicas = False"
    assert lines[2] == "num_workers=2: dos pasadas identicas = True"


def test_kaggle_worker_setting_is_recorded():
    text = (FOLDER / "kaggle_config_output.txt").read_text(encoding="utf-8")
    assert "KAGGLE" in text and "NUM_WORKERS       : 2" in text


def test_readme_points_to_the_evidence():
    for ref in ("results/dataloader_rng_check/output.txt", "scripts/check_dataloader_rng.py",
                "results/dataloader_rng_check/kaggle_config_output.txt", "0.8415 and 0.8427"):
        assert ref in README, ref


def test_check_script_defines_the_same_experiment():
    src = (ROOT / "scripts" / "check_dataloader_rng.py").read_text()
    assert "num_workers=workers" in src and "for workers in (0, 2)" in src
