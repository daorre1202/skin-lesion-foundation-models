# First training session, 31 July 2026

`stdout_from_notebook.txt` is the output that was stored in the notebook after the first training session. It is included as evidence for the provenance of the checkpoints (see the main README).

It was produced by an older version of the notebook whose ensemble stage used a fixed list of three models, so the ensemble lines at the end of the file (0.8266, 0.8350, 0.8420) are not results of this repository. The per-model lines are the ones used: ResNet-50, DenseNet-121, EfficientNet-B3 and DINOv2 match the canonical run, and ViT-B/16 and BiomedCLIP show the results with the loss setting that was later replaced.
