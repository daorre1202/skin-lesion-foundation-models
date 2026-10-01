"""Re-run inference for the six checkpoints and save per-model probabilities.

Run inside WSL with the CE1 environment and a GPU. For every model it saves, for the
validation and the test split, the softmax probabilities without TTA (raw) and the mean of
10 TTA rounds (tta). Images are loaded in the same order the training notebook used, and
that order is checked against the labels saved in the canonical run.

TTA uses the same transforms as the notebook (random rotation by a multiple of 90 degrees,
horizontal and vertical flips, each with probability 0.5), drawn from a seeded generator,
so a rerun gives the same arrays. The notebook's TTA ran through DataLoader workers, which
probably repeated the same views in every round (see scripts/check_dataloader_rng.py), so
the TTA numbers here differ from the canonical run.

Example:
    python scripts/reinfer_per_model.py \\
        --images ~/casesys/data/raw/ham10000/images \\
                 ~/ce1-data/ISIC2018_Task3_Validation_Input \\
                 ~/ce1-data/ISIC2018_Task3_Test_Input \\
        --labels-dir data/ground_truth --checkpoints ~/ce1-data/checkpoints
Quick check before the full run: add  --limit 64 --models resnet50 vit_b16 --rounds 2 --out /tmp/ce1_smoke
"""
import argparse
import json
import platform
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.ce1_common import (IMG_SIZE, MODELS, ROOT, RUN, index_images, label_array,  # noqa: E402
                                load_labels, ordered_ids)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
TOLERANCE = 0.002      # allowed gap between recomputed and stored BACC
ABORT_GAP = 0.02       # a larger gap means the preprocessing or the weights are wrong


def build_model(name, num_classes=7):
    """Same architectures as the training notebook, without downloading pretrained weights."""
    import open_clip
    import timm
    from torchvision.models import densenet121, efficientnet_b3, resnet50

    native = None
    if name == "resnet50":
        m = resnet50(weights=None)
        m.fc = nn.Linear(m.fc.in_features, num_classes)
    elif name == "densenet121":
        m = densenet121(weights=None)
        m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    elif name == "efficientnet_b3":
        m = efficientnet_b3(weights=None)
        m.classifier[1] = nn.Linear(m.classifier[1].in_features, num_classes)
    elif name == "vit_b16":
        m = timm.create_model("vit_base_patch16_224.augreg2_in21k_ft_in1k",
                              pretrained=False, num_classes=num_classes)
        native = {"mean": list(m.pretrained_cfg["mean"]), "std": list(m.pretrained_cfg["std"])}
    elif name == "dinov2_b":
        m = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=False,
                              num_classes=num_classes, img_size=IMG_SIZE["dinov2_b"])
        native = {"mean": list(m.pretrained_cfg["mean"]), "std": list(m.pretrained_cfg["std"])}
    elif name == "biomedclip":
        clip_model, _, preprocess = open_clip.create_model_and_transforms(
            "hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224")

        class BiomedCLIPClassifier(nn.Module):
            def __init__(self, visual, n):
                super().__init__()
                self.visual = visual
                with torch.no_grad():
                    dim = self.visual(torch.zeros(1, 3, 224, 224)).shape[-1]
                self.head = nn.Linear(dim, n)

            def forward(self, x):
                return self.head(self.visual(x))

        m = BiomedCLIPClassifier(clip_model.visual, num_classes)
        for t in getattr(preprocess, "transforms", []):
            if hasattr(t, "mean") and hasattr(t, "std"):
                native = {"mean": [float(v) for v in t.mean], "std": [float(v) for v in t.std]}
    else:
        raise ValueError(f"unknown model: {name}")
    if native is None:
        native = {"mean": IMAGENET_MEAN, "std": IMAGENET_STD}
    return m, native


def load_resized(paths, size, workers=8):
    """Decode and resize with albumentations, as the notebook did. Returns uint8 (N,3,S,S)."""
    resize = A.Resize(size, size)

    def one(path):
        return resize(image=np.array(Image.open(path).convert("RGB")))["image"]

    with ThreadPoolExecutor(workers) as pool:
        arrays = list(pool.map(one, paths))
    return torch.from_numpy(np.stack(arrays)).permute(0, 3, 1, 2).contiguous()


def tta_views(x, gen):
    """Random rot90 (p=0.5, factor 0-3), horizontal flip (p=0.5), vertical flip (p=0.5)."""
    b = x.shape[0]
    rot = (torch.rand(b, generator=gen) < 0.5) * torch.randint(0, 4, (b,), generator=gen)
    hflip = torch.rand(b, generator=gen) < 0.5
    vflip = torch.rand(b, generator=gen) < 0.5
    out = x.clone()
    for k in (1, 2, 3):
        idx = (rot == k).nonzero(as_tuple=True)[0].to(x.device)
        if len(idx):
            out[idx] = torch.rot90(x[idx], k, dims=(2, 3))
    for dim, mask in ((3, hflip), (2, vflip)):
        idx = mask.nonzero(as_tuple=True)[0].to(x.device)
        if len(idx):
            out[idx] = torch.flip(out[idx], dims=(dim,))
    return out


@torch.no_grad()
def predict(model, data, device, mean, std, batch, gen=None):
    out = []
    for start in range(0, len(data), batch):
        x = data[start:start + batch].to(device, non_blocking=True).float().div_(255.0)
        x = (x - mean) / std
        if gen is not None:
            x = tta_views(x, gen)
        out.append(torch.softmax(model(x), dim=1).float().cpu())
    return torch.cat(out).numpy()


def predict_tta(model, data, device, mean, std, batch, rounds, seed):
    acc = None
    for r in range(rounds):
        gen = torch.Generator()
        gen.manual_seed(seed + r)
        p = predict(model, data, device, mean, std, batch, gen)
        acc = p if acc is None else acc + p
    return acc / rounds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--checkpoints", required=True)
    ap.add_argument("--out", default=str(ROOT / "results" / "reinference_2026-10"))
    ap.add_argument("--models", nargs="+", default=MODELS, choices=MODELS)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: use about N images per split")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu" and not args.limit:
        raise SystemExit("No GPU found. Use --limit for a CPU smoke test.")
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

    assignment = json.loads((RUN / "split_assignment.json").read_text())
    labels = load_labels(args.labels_dir)
    ids = {s: ordered_ids(labels, assignment, s) for s in ("val", "test")}
    reference = {"val": np.load(RUN / "tta_val_labels.npy"), "test": np.load(RUN / "tta_labels.npy")}
    for s in ids:
        if not (label_array(labels, ids[s]) == reference[s]).all():
            raise SystemExit(f"Image order of the {s} split does not match the canonical labels.")
    print(f"order check passed: val {len(ids['val'])}, test {len(ids['test'])}")

    index = index_images(args.images)
    missing = [i for s in ids.values() for i in s if i not in index]
    if missing:
        raise SystemExit(f"{len(missing)} images not found, for example {missing[:5]}")
    if args.limit:
        for s in ids:
            step = max(1, len(ids[s]) // args.limit)
            keep = list(range(0, len(ids[s]), step))[:args.limit]
            ids[s] = [ids[s][i] for i in keep]
            reference[s] = reference[s][keep]
    labels_y = {s: reference[s] for s in ids}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sizes = sorted({IMG_SIZE[m] for m in args.models})
    cache = {}
    for size in sizes:
        t0 = time.time()
        cache[size] = {s: load_resized([index[i] for i in ids[s]], size) for s in ids}
        print(f"decoded images at {size}px in {time.time() - t0:.0f}s")

    meta_checks, report = {}, {}
    for name in args.models:
        t0 = time.time()
        model, native = build_model(name)
        state = torch.load(Path(args.checkpoints) / f"{name}_best.pth", map_location="cpu", weights_only=True)
        if isinstance(state, dict) and isinstance(state.get("model"), dict):
            state = state["model"]
        model.load_state_dict(state, strict=True)
        model.to(device).eval()
        mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
        stored = json.loads((RUN / f"{name}_meta.json").read_text())
        report[name] = {"normalization_used": {"mean": IMAGENET_MEAN, "std": IMAGENET_STD},
                        "normalization_native": native}
        for split in ("val", "test"):
            data = cache[IMG_SIZE[name]][split]
            raw = predict(model, data, device, mean, std, args.batch)
            tta = predict_tta(model, data, device, mean, std, args.batch, args.rounds, args.seed)
            np.save(out / f"probs_{name}_{split}_raw.npy", raw.astype(np.float32))
            np.save(out / f"probs_{name}_{split}_tta.npy", tta.astype(np.float32))
            b_raw = balanced_accuracy_score(labels_y[split], raw.argmax(1))
            b_tta = balanced_accuracy_score(labels_y[split], tta.argmax(1))
            report[name][split] = {"bacc_raw": round(float(b_raw), 4), "bacc_tta": round(float(b_tta), 4),
                                   "bacc_stored": round(stored[f"{split}_bacc"], 4)}
        if not args.limit:
            gaps = {s: report[name][s]["bacc_raw"] - report[name][s]["bacc_stored"] for s in ("val", "test")}
            status = "OK" if all(abs(g) <= TOLERANCE for g in gaps.values()) else "DIFF"
            report[name]["gap_vs_stored"] = {k: round(v, 4) for k, v in gaps.items()}
            report[name]["status"] = status
            print(f"{name:16s} val {report[name]['val']['bacc_raw']:.4f} (stored {report[name]['val']['bacc_stored']:.4f})  "
                  f"test {report[name]['test']['bacc_raw']:.4f} (stored {report[name]['test']['bacc_stored']:.4f})  "
                  f"{status}  {time.time() - t0:.0f}s")
            if max(abs(g) for g in gaps.values()) > ABORT_GAP:
                raise SystemExit(f"{name}: gap above {ABORT_GAP}. Check preprocessing and weights before going on.")
        else:
            print(f"{name:16s} smoke test done in {time.time() - t0:.0f}s")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    import open_clip
    import timm
    info = {"device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "timm": timm.__version__, "open_clip": open_clip.__version__, "albumentations": A.__version__,
            "tta_rounds": args.rounds, "tta_seed": args.seed, "batch": args.batch,
            "smoke_test": bool(args.limit), "models": report}
    (out / "reinference_meta.json").write_text(json.dumps(info, indent=2))
    print(f"saved to {out}")


if __name__ == "__main__":
    main()
