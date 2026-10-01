"""Do two passes over a DataLoader give the same random augmentations?

Albumentations keeps its own random generator inside each transform. A DataLoader with
workers copies the dataset, and so the generator, into every worker at the start of every
pass, so each pass can draw the same augmentations again. With zero workers the generator
lives in the main process and advances between passes.

Run in the environment of the re-inference:
    python scripts/check_dataloader_rng.py
"""
import albumentations as A
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset


class Images(Dataset):
    def __init__(self):
        rng = np.random.default_rng(0)
        self.imgs = [rng.integers(0, 255, (16, 16, 3), dtype=np.uint8) for _ in range(64)]
        self.t = A.Compose([A.RandomRotate90(p=0.5), A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.5)])

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, i):
        return torch.from_numpy(self.t(image=self.imgs[i])["image"].copy())


if __name__ == "__main__":
    ds = Images()
    print("albumentations", A.__version__)
    for workers in (0, 2):
        loader = DataLoader(ds, batch_size=8, shuffle=False, num_workers=workers)
        a = torch.cat(list(loader))
        b = torch.cat(list(loader))
        print(f"num_workers={workers}: dos pasadas identicas = {bool((a == b).all())}")
