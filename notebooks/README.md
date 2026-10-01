# Notebook map

`ce1_pipeline.ipynb` is the pipeline as it ran. Comments and log messages are in Spanish. This table maps the section headers.

| Section | Content |
|---|---|
| 00 Installation | `pip install albumentations timm open_clip_torch` |
| 01 Imports | Libraries, including timm (ViT, DINOv2) and open_clip (BiomedCLIP) |
| 02 Reproducibility and seed | Seed 42 for Python, NumPy and PyTorch, deterministic cuDNN |
| (unlabelled) | Copies the checkpoint folder from a Kaggle dataset to the working directory |
| 03 Global configuration | Classes, resolution, batch size, learning rates, early stopping, TTA rounds, per-model regime (warmup, linear probing, focal gamma) |
| 04 Environment detection | Colab, Kaggle or local paths |
| 05 Execution modes | `LOAD_TTA_FROM_DIR`, `RESUME_FROM_CHECKPOINTS`, `RESUME_DIR`, `FORCE_RETRAIN` |
| 06 Augmentation and clinical thresholds | Augmentation boost for MEL, threshold targets for MEL and AKIEC |
| 07 Logging | Tee of stdout to `ejecucion_log.txt` |
| 08 Ground truth and image index | Merges the three ISIC CSV files into one table |
| 09 Stratified split | Deterministic 60 / 20 / 20 split, saved as `split_assignment.json` |
| 10 to 13 | Dataset with graded augmentation, transforms, focal loss with label smoothing, weighted sampler |
| 14 Build model | Six backbones and their classification heads |
| 15 Early stopping | Mixed score of BACC and validation loss |
| 16 Train and evaluate | Training loop with gradient clipping |
| 17 TTA | Geometric test-time augmentation |
| 18 Grad-CAM | Used only for the PDF report |
| 19 Directory preparation | Copies images to local storage |
| 20 Threshold calibration | `calibrate_threshold`, `apply_clinical_thresholds`, per-class metrics |
| 21 Main | Loads or trains the six models, builds the ensembles, searches the best subset on validation, calibrates thresholds, writes the result files |
| 22 PDF report | Builds the Spanish PDF report, which is not distributed |
| 23 Entry point | Runs `main()` |
