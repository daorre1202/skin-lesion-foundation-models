# Foundation models and CNN ensembles for dermoscopic classification on ISIC 2018

![CI](https://github.com/daorre1202/skin-lesion-foundation-models/actions/workflows/ci.yml/badge.svg)
![License](https://img.shields.io/badge/License-MIT-green?style=flat)

Follow-up to my [undergraduate thesis](https://github.com/daorre1202/skin-lesion-classifier-CNN) on the same dataset and the same split as its seed-42 reference run (ISIC 2018 Task 3, HAM10000; 7,030 / 2,342 / 2,348 images for train / validation / test). Three ImageNet CNNs (ResNet-50, DenseNet-121, EfficientNet-B3) are compared with three transformer backbones (ViT-B/16, DINOv2 ViT-B/14, BiomedCLIP) under full fine-tuning, validation-weighted ensembling, 10-round test-time augmentation (TTA) and per-class decision thresholds for melanoma and actinic keratosis.

**Research code. It is not a medical device, it has no clinical validation and it must not be used to diagnose or to decide on any treatment.**

## In short

- Combining the six models beats the three CNNs by +0.0132 BACC (95% interval +0.0023 to +0.0244). The data do not show that foundation models beat CNNs.
- 39% of the HAM10000 test images share a lesion with train or validation. Without them the canonical ensemble scores 0.804 BACC, against 0.8415 on the full test set.
- Picking the best subset of models on validation overfits: the picked subsets lose 0.0234 and 0.0337 BACC on test.
- The canonical TTA probably repeated one random view in all rounds, so the re-inference with ten distinct views is the reference for TTA.

The canonical run is a single training run with seed 42. Its checkpoints come from more than one session, and one hyperparameter was chosen by looking at test BACC. A later re-inference with the same weights adds confidence intervals. Read [Provenance and limitations](#provenance-and-limitations) before using any number.

## What the data show

1. **The split shares lesions between train, validation and test, and it inflates every figure here.** HAM10000 holds several images of some lesions, and the split is by image. Of the 1,991 HAM10000 test images, 779 (39%) have another image of the same lesion in train or validation, and 577 of the 2,000 HAM10000 validation images (29%) have one in train. The other 357 test images come from the official ISIC 2018 validation and test sets, which have no lesion identifier. On the 1,212 test images without a lesion-mate in train or validation, the canonical ensemble scores 0.8041 BACC (95% interval 0.744 to 0.862) and 0.8095 with thresholds, against 0.8415 and 0.8499 on the full test set. Malignant BACC goes from 0.7102 with argmax to 0.7466 with thresholds, a gain of +0.0364 (interval +0.0106 to +0.0681), so the effect of the thresholds survives. The seed-42 reference run of the thesis repository uses the same split, so the same overlap applies to it. Details [below](#lesion-overlap-between-splits).
2. **Choosing the best subset of models on validation overfits.** Of the 63 possible subsets, the canonical run picks ResNet-50 + ViT-B/16 + DINOv2: 0.8613 on validation, 0.8379 on test, a fall of 0.0234 (the six-model ensemble ranks fifth of 63 on validation, with 0.8552). The re-inference, with the same weights and different TTA draws, picks another subset (five models, without EfficientNet-B3): 0.8661 on validation, 0.8323 on test, a fall of 0.0337. In the re-inference the 63 subsets lose 0.0093 on average between validation and test, because checkpoints and weights were also chosen on validation, so both picks lose more than average. In both runs the picked subset scores below the six-model ensemble on test (0.8379 and 0.8323 against 0.8415). The effect is not uniform: the subset ranked second on validation in the re-inference, which is the canonical pick, loses only 0.0094 and reaches 0.8546 on test. With 2,348 test images and classes of 32 and 36 images, one selection cannot tell which subset is best.
3. **Combining the six models beats the three CNNs by a small margin. The data do not show that foundation models beat CNNs.** Six models reach 0.8415 against 0.8283 for the three CNNs in the canonical run (+0.0132) and 0.8427 against 0.8295 in the re-inference (+0.0132, 95% interval +0.0023 to +0.0244). The three foundation models alone against the three CNNs give +0.0087 in the canonical run and +0.0068 in the re-inference, with an interval of -0.0168 to +0.0288. The six-against-three contrast also changes the number of ensemble members. On the 1,212 test images without lesion overlap the six-against-three difference is +0.0111 (-0.0069 to +0.0308) and the foundation-against-CNN difference is -0.0055 (-0.0335 to +0.0233).
4. **Only the TTA gain of DINOv2 reproduces.** DINOv2 gains +0.0236 in the canonical run and +0.0239 in the re-inference (interval +0.0095 to +0.0399). For the other five models the two runs disagree by up to 0.0165 (see the table below), and at ensemble level the gain is +0.0055 with an interval of -0.0100 to +0.0213. The canonical TTA probably repeated one random view in all 10 rounds (see [Provenance](#provenance-and-limitations)), so the re-inference figures are the ones to use for TTA.

## Results of the canonical run

Test set, 2,348 images. BACC is balanced accuracy. All figures are stored in [`results/canonical_run_2026-08-04`](results/canonical_run_2026-08-04) and recomputed by the tests where the saved arrays allow it.

### Individual models

| Model | Val BACC | Test BACC | Test BACC with TTA | TTA gain | Best epoch / epochs run | Training time |
|---|---|---|---|---|---|---|
| ResNet-50 | 0.7977 | 0.7881 | 0.7934 | +0.0053 | 34 / 35 | 58 min |
| DenseNet-121 | 0.7952 | 0.7862 | 0.7902 | +0.0040 | 30 / 35 | 51 min |
| EfficientNet-B3 | 0.7975 | 0.7940 | 0.7865 | -0.0075 | 11 / 21 | 58 min |
| ViT-B/16 | 0.8259 | 0.8009 | 0.8011 | +0.0002 | 8 / 18 | 88 min |
| DINOv2 ViT-B/14 | 0.8013 | 0.7950 | 0.8187 | +0.0236 | 25 / 35 | 193 min |
| BiomedCLIP | 0.7949 | 0.7882 | 0.7802 | -0.0081 | 31 / 35 | 171 min |

The six checkpoints add up to 618 minutes of training, produced in different sessions (see below).

### Ensembles

Weights are proportional to validation BACC.

| Ensemble | Test BACC |
|---|---|
| Six models, no TTA | 0.8371 |
| Three CNNs, 10-round TTA | 0.8283 |
| Three foundation models, 10-round TTA | 0.8370 |
| **Six models, 10-round TTA (official ensemble)** | **0.8415** |
| Six models, 10-round TTA, clinical thresholds | 0.8499 |

In the canonical run the ten TTA rounds probably used the same random view (see [Provenance](#provenance-and-limitations)). The re-inference below uses ten different views.

### Clinical thresholds

Thresholds are calibrated on validation only: the highest value of a 199-point grid that meets the sensitivity target and the specificity floor (MEL: sensitivity 0.85 and specificity 0.85; AKIEC: sensitivity 0.75 and specificity 0.70). The result is theta(MEL) = 0.307 and theta(AKIEC) = 0.386.

| | Argmax | With thresholds |
|---|---|---|
| Global BACC | 0.8415 | 0.8499 (+0.0084) |
| Malignant BACC (mean sensitivity of MEL, BCC, AKIEC) | 0.7861 | 0.8250 (+0.0388) |

The malignant gain is computed on unrounded values. Sensitivity and specificity per class on the test set:

| Class | Sensitivity, argmax | Sensitivity, thresholds | Specificity, thresholds |
|---|---|---|---|
| MEL | 0.7241 | 0.8276 | 0.9492 |
| NV | 0.9457 | 0.9251 | 0.9225 |
| BCC | 0.9200 | 0.9200 | 0.9879 |
| AKIEC | 0.7143 | 0.7273 | 0.9934 |
| BKL | 0.8364 | 0.7993 | 0.9822 |
| DF | 0.7500 | 0.7500 | 0.9983 |
| VASC | 1.0000 | 1.0000 | 0.9987 |

At the calibrated thresholds the one-vs-rest sensitivity is 0.8506 for MEL and 0.7733 for AKIEC on validation, which meets the targets (0.85 and 0.75), and 0.8238 and 0.7143 on test, which misses them. The per-class table above comes from the final multi-class decision and gives 0.8276 and 0.7273. The thresholds also cost sensitivity on the two largest benign classes: NV from 0.9457 to 0.9251 and BKL from 0.8364 to 0.7993.

## Re-inference with confidence intervals

The canonical run did not save per-model probabilities. In October 2026 the six checkpoints were run again with `scripts/reinfer_per_model.py` (details in [`results/reinference_2026-10`](results/reinference_2026-10)). Without TTA the six models reproduce the stored test BACC to four decimals and the stored validation BACC to within 0.0006. TTA is drawn from a seeded generator, so its numbers differ from the canonical run.

Intervals come from a paired bootstrap over the 2,348 test images (10,000 resamples, seed 42, the same resample for every contrast).

| Contrast, test BACC | Difference | 95% interval | Resamples with difference > 0 |
|---|---|---|---|
| Six models against three CNNs, TTA | +0.0132 | +0.0023 to +0.0244 | 0.9917 |
| Three foundation models against three CNNs, TTA | +0.0068 | -0.0168 to +0.0288 | 0.7230 |
| DINOv2 with TTA against without | +0.0239 | +0.0095 to +0.0399 | 0.9996 |
| Six models with TTA against without | +0.0055 | -0.0100 to +0.0213 | 0.7611 |

Ensembles in the re-inference: three CNNs 0.8295, three foundation models 0.8363, six models 0.8427 with TTA and 0.8371 without.

TTA gain per model, test BACC:

| Model | Without TTA | Re-inference with TTA | Gain, re-inference | Gain, canonical run |
|---|---|---|---|---|
| ResNet-50 | 0.7881 | 0.7986 | +0.0105 | +0.0053 |
| DenseNet-121 | 0.7862 | 0.8048 | +0.0186 | +0.0040 |
| EfficientNet-B3 | 0.7940 | 0.8029 | +0.0089 | -0.0075 |
| ViT-B/16 | 0.8009 | 0.8134 | +0.0125 | +0.0002 |
| DINOv2 ViT-B/14 | 0.7950 | 0.8190 | +0.0239 | +0.0236 |
| BiomedCLIP | 0.7882 | 0.7966 | +0.0084 | -0.0081 |

The cause of the disagreement between the two columns is not established. Per-model BACC moves by several thousandths when a handful of images in the 32-image and 36-image classes change.

Subset search in the re-inference. For each model the representation that scored higher on validation is used (TTA for ResNet-50, EfficientNet-B3, ViT-B/16 and DINOv2, no TTA for DenseNet-121 and BiomedCLIP), as in the canonical run:

| Rank on validation | Subset | Val BACC | Test BACC |
|---|---|---|---|
| 1 | ResNet-50, DenseNet-121, ViT-B/16, DINOv2, BiomedCLIP | 0.8661 | 0.8323 |
| 2 | ResNet-50, ViT-B/16, DINOv2 | 0.8640 | 0.8546 |
| 7 | All six | 0.8590 | 0.8415 |

## Lesion overlap between splits

`scripts/lesion_overlap.py` flags every validation and test image whose lesion also appears in a split it should not share (validation against train, test against train and validation), using `HAM10000_metadata.csv`. The flags are stored as identifiers only in [`results/lesion_overlap/flags.csv`](results/lesion_overlap/flags.csv); the metadata file is not distributed. Test images by group:

| Group | Images | Argmax BACC | BACC with thresholds | Malignant BACC, argmax to thresholds |
|---|---|---|---|---|
| All test images | 2,348 | 0.8415 | 0.8499 | 0.7861 to 0.8250 |
| No lesion-mate in train or validation | 1,212 | 0.8041 | 0.8095 | 0.7102 to 0.7466 |
| Lesion-mate in train or validation | 779 | 0.8782 | 0.8845 | 0.8496 to 0.8846 |
| No lesion identifier (official ISIC 2018 sets) | 357 | 0.8360 | 0.8464 | 0.8056 to 0.8506 |

The figures are for the canonical six-model TTA ensemble. The 95% interval of the BACC without lesion-mate is 0.744 to 0.862 with argmax and 0.750 to 0.866 with thresholds (bootstrap over the 1,212 images). The gain of the thresholds on malignant BACC is +0.0388 on all images (+0.0251 to +0.0547), +0.0364 without lesion-mate (+0.0106 to +0.0681) and +0.0350 with one (+0.0201 to +0.0516).

The groups also differ in composition, so these figures do not isolate the effect of the overlap. The group without lesion-mate has 62 MEL, 943 NV, 41 BCC, 35 AKIEC, 111 BKL, 9 DF and 11 VASC images, which makes its BACC noisy, and its interval contains the value of the full test set. The measurement is a lower bound: the 357 images without lesion identifier were not checked, and a lesion can also appear under two identifiers. The thresholds, the ensemble weights and the checkpoints were chosen on a validation set where 29% of the HAM10000 images have a lesion-mate in train.

## Provenance and limitations

- **One training run, one seed.** In the thesis repository the same CNN ensemble varies by 0.009 (standard deviation) across three seeds, but each of those seeds also draws its own partition, so the spread for the fixed split used here is not known. The intervals above cover the sampling of the test images for fixed weights and do not cover the variability of training.
- **The canonical TTA probably repeated one view.** The notebook builds its TTA loader with `num_workers=NUM_WORKERS` and no `worker_init_fn`, and the stored output of its configuration cell shows `NUM_WORKERS : 2` on Kaggle ([`kaggle_config_output.txt`](results/dataloader_rng_check/kaggle_config_output.txt)). With albumentations 2.0.8, a PyTorch DataLoader with two workers gives the same random augmentations on every pass over the dataset, and with zero workers it does not ([`scripts/check_dataloader_rng.py`](scripts/check_dataloader_rng.py), output in [`results/dataloader_rng_check`](results/dataloader_rng_check/output.txt)). The albumentations version of the Kaggle sessions was not recorded, because the installation cell does not pin it. If the behaviour applied there, each canonical TTA round is the same flip and rotation of every image. That fits the per-model gains that disagree with the re-inference, whose ten views differ, while the ensemble figures agree (0.8415 and 0.8427). The training loaders sample images in shuffled order, so one image is unlikely to receive the same draw in every epoch; that was not measured.
- **Checkpoints from different sessions.** ResNet-50, DenseNet-121, EfficientNet-B3 and DINOv2 were trained from scratch on 31 July 2026; their validation BACC, test BACC, best epoch and training time match the output stored in [`results/earlier_session_2026-07-31`](results/earlier_session_2026-07-31/stdout_from_notebook.txt). ViT-B/16 and BiomedCLIP come from a later session with a different loss setting (below). On 4 August 2026 the six checkpoints were evaluated together by resuming from one checkpoint folder: the log in the canonical run says `REANUDAR` (resume) and no model was trained in that run. The log of the session that produced the later two checkpoints is not part of this repository.
- **The loss setting of the foundation models was chosen on test BACC.** The notebook comment next to `FOCAL_GAMMA` records that gamma = 0 and gamma = 1 were compared by test BACC: ViT-B/16 and BiomedCLIP improved with gamma = 1 (+0.0054 and +0.0443) and DINOv2 got worse (-0.0141) and keeps gamma = 0. The first two differences match the stored numbers (ViT-B/16 0.7955 to 0.8009, BiomedCLIP 0.7439 to 0.7882). Validation points the same way (ViT-B/16 0.8185 to 0.8259, BiomedCLIP 0.7734 to 0.7949), but the gamma = 1 run of DINOv2 was not kept, so its validation score is unknown. Test figures of the foundation models, and of any ensemble that includes them, can be biased upwards. The three-CNN ensemble is not affected.
- **Normalisation.** All six models were trained and evaluated with the ImageNet mean and standard deviation. The pretrained configuration of ViT-B/16 uses 0.5 and 0.5 for both, and BiomedCLIP uses the CLIP constants (mean 0.481, 0.458, 0.408; standard deviation 0.269, 0.261, 0.276). DINOv2 uses the ImageNet values. The effect of the mismatch was not measured.
- **The CNN baseline is lower than in the thesis.** The thesis TTA ensemble of the same three architectures reached 0.8486 for seed 42 on the same split, and 0.8361 to 0.8545 across three seeds with their own partitions. Here it reaches 0.8283. The cause was not investigated, so the +0.0132 above is relative to this lower baseline.
- **Not documented or not covered.** The learning rate of 1e-5 for the transformers, the three-epoch warmup of ViT-B/16 and BiomedCLIP, and the 10-epoch linear-probing phase of DINOv2 are fixed in the configuration cell and how they were chosen is not recorded. AKIEC has 75 validation and 77 test images, so one image moves its sensitivity by about 1.3 points. One dataset, no external validation, no claim about foundation models in general.

## Dataset

ISIC 2018 Challenge Task 3 (HAM10000), seven classes, 11,720 images: the 10,015 training images, the 193 validation images and the 1,512 test images of the challenge, merged and split again with a stratified 60 / 20 / 20 split (seed 42). HAM10000 was collected at two sites, in Vienna and in Queensland.

| Class | Train | Validation | Test |
|---|---|---|---|
| MEL melanoma | 783 | 261 | 261 |
| NV melanocytic nevus | 4,642 | 1,547 | 1,548 |
| BCC basal cell carcinoma | 373 | 124 | 125 |
| AKIEC actinic keratosis / intraepithelial carcinoma | 226 | 75 | 77 |
| BKL benign keratosis-like lesion | 802 | 267 | 269 |
| DF dermatofibroma | 96 | 32 | 32 |
| VASC vascular lesion | 108 | 36 | 36 |

Nevi are 66% of the images. The images are not included here. The data are distributed under CC BY-NC 4.0 and must be downloaded from the ISIC challenge site. `split_assignment.json` lists image identifiers only.

## Reproduce and verify

Verification runs on CPU in seconds and needs only the files in this repository:

```
python -m pip install -r requirements.txt
python scripts/verify_results.py
python -m pytest
```

`scripts/verify_results.py` recomputes the canonical ensemble BACC, recalibrates both thresholds from the validation arrays, rebuilds the per-class tables and compares everything with the stored values. The tests also check the split counts, the numbers quoted in the tables and findings of this README against the stored files, the consistency of the confusion tables, the re-inference arrays, the bootstrap analysis, the lesion-overlap analysis, the notebook code hash and the SHA-256 hashes in [`results/canonical_run_2026-08-04/SHA256SUMS.txt`](results/canonical_run_2026-08-04/SHA256SUMS.txt).

The re-inference needs a GPU, the images, the ground-truth CSV files and the six checkpoints (about 1.2 GB; the checkpoints are not distributed):

```
python -m pip install -r requirements-reinference.txt
python scripts/reinfer_per_model.py --images <image folders> --labels-dir <ground-truth folder> --checkpoints <folder with *_best.pth>
python -m scripts.analyze_reinference
```

Training code is in [`notebooks/ce1_pipeline.ipynb`](notebooks/ce1_pipeline.ipynb), executed on Kaggle. Its code is as it ran (only two inaccurate comments were corrected, and a test checks that the code tokens did not change). Comments and log messages are in Spanish, and a cell-by-cell map in English is in [`notebooks/README.md`](notebooks/README.md). The configuration cell is set to resume from the hybrid checkpoint folder, which is not distributed. A fresh training run needs `RESUME_FROM_CHECKPOINTS = False`, about ten hours of GPU time, and gives different numbers.

## Repository layout

```
notebooks/                          executed pipeline (outputs cleared) and its English cell map
results/canonical_run_2026-08-04/   files of the canonical run, unmodified, with hashes and a glossary
results/earlier_session_2026-07-31/ stdout stored in the notebook from the first training session
results/reinference_2026-10/        per-model probabilities and the bootstrap analysis
results/lesion_overlap/             lesion-overlap flags (identifiers only) and the subset analysis
results/dataloader_rng_check/       output of the DataLoader check and the Kaggle worker setting
scripts/                            verification, re-inference and analysis
tests/                              pytest suite run by CI
```

## Licence and citation

The code is released under the MIT licence. See [`CITATION.cff`](CITATION.cff).
