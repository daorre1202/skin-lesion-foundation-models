# Glossary of the canonical run

The files in this folder are the originals of the run of 4 August 2026, unmodified (hashes in `SHA256SUMS.txt`). File names and column headers are Spanish, as the notebook wrote them.

Class order everywhere: MEL, NV, BCC, AKIEC, BKL, DF, VASC.

| File | Content |
|---|---|
| `results.json` | Machine-readable summary (English keys): run settings, per-model validation and test BACC, official ensemble, thresholds and per-class metrics |
| `resumen_modelos.csv` | Per-model table. `Mejor epoch` best epoch, `Epochs totales` epochs run, `Tiempo` training time. The last rows give the ensemble results |
| `comparativa_ensembles.csv` | Test BACC of the ensembles. `CNN (3 originales)` three CNNs, `Fundacionales (3 nuevos)` three foundation models, `Todos` all six, `Mejor subconjunto (val)` best subset chosen on validation |
| `ganancia_tta_por_modelo.csv` | Test BACC without and with TTA per model (`Ganancia TTA` is the difference) |
| `busqueda_subconjuntos.csv` | Validation BACC of the 63 model subsets. `Modelos` models in the subset, `N` their number |
| `tta_decision_por_modelo.csv` | Validation BACC without and with TTA per model and which one won (`Usa_TTA`) |
| `metricas_clinicas_umbrales_clinicos.csv` | Test confusion counts and per-class sensitivity (`Sensibilidad`) and specificity (`Especificidad`) of the final system: six-model TTA ensemble with clinical thresholds |
| `metricas_clinicas_vit_b16.csv` | The same table for the best individual model, ViT-B/16 without TTA, argmax |
| `*_meta.json`, `*_history.json` | Per model: best validation BACC and loss, test BACC, best epoch and training seconds; per-epoch training and validation curves |
| `tta_sum_probs.npy`, `tta_labels.npy` | Test set: class probabilities of the six-model TTA ensemble (2,348 x 7, rows sum to 1) and labels |
| `tta_val_sum.npy`, `tta_val_labels.npy` | The same for the validation set (2,342 images) |
| `tta_thresh_preds.npy` | Test predictions after applying the clinical thresholds |
| `calibrated_thresholds.json` | Calibrated thresholds for MEL and AKIEC |
| `class_counts.json` | Images per class as [train, validation, test] |
| `classes_used.json` | Class order |
| `split_assignment.json` | Image identifier to split (identical to the thesis split) |
| `ejecucion_log.txt` | Original log of the run, in Spanish. The mode line `REANUDAR` shows that the run resumed from stored checkpoints |

The last line of the log names the report `resultados_tfg.pdf`, because the notebook was derived from the thesis pipeline. The file was renamed `resultados_ce1.pdf` afterwards, since this case study is not part of the thesis. The PDF is in Spanish and is not distributed; every number in it is also in the files above.
