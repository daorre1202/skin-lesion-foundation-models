# Re-inference of the six checkpoints, October 2026

The six checkpoints of the canonical run were loaded again and run on the validation and test splits with `scripts/reinfer_per_model.py`. The purpose is to have per-model probabilities, which the canonical run did not save, so that confidence intervals and the subset search can be computed.

| File | Content |
|---|---|
| `probs_<model>_<split>_raw.npy` | Softmax probabilities without TTA, float32, shape (images, 7). `<split>` is `val` (2,342 images) or `test` (2,348) |
| `probs_<model>_<split>_tta.npy` | Mean of 10 TTA rounds |
| `reinference_meta.json` | Hardware, library versions, TTA settings, normalisation constants and the comparison of each model with its stored BACC |
| `analysis.json` | Output of `python -m scripts.analyze_reinference`: ensembles, paired bootstrap, subset search |

Rows follow the order of the training notebook (classes in the order MEL, NV, BCC, AKIEC, BKL, DF, VASC, file names sorted inside each class). The script checks that order against the labels saved in the canonical run before it does anything else.

## Differences with the canonical run

- Same weights, same split, same preprocessing (albumentations `Resize`, ImageNet mean and standard deviation). Hardware differs: RTX 5070 Ti here, a Kaggle GPU in the canonical run.
- Without TTA the six models reproduce the stored test BACC to four decimals. The validation BACC differs by at most 0.0006.
- TTA uses the same random transforms as the notebook (rotation by a multiple of 90 degrees, horizontal flip, vertical flip, each with probability 0.5) but drawn from a seeded generator (seed 2026 plus the round number) and applied to batches on the GPU. Every model sees the same transforms, and the ten rounds use ten different views. The canonical run probably repeated one view in all rounds (see the main README), so its TTA numbers are not comparable.
- `analysis.json` was regenerated with the scripts of this repository from the arrays and checked against the first output of the run, which it matches.
