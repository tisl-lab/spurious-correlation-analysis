"""
datasets/colored_mnist.py
=========================

Colored MNIST Dataset Loader
-----------------------------

Loads the pre-colored Colored MNIST dataset (colorized-MNIST by jayaneetha/youssifhisham),
where each MNIST image has a solid red, green, or blue background.

The dataset already exists on disk — no data generation or modification is done here.
Colors are randomly assigned in the original dataset (each digit appears in all 3 colors
in roughly equal proportions), which lets the analysis reveal whether CLIP's zero-shot
predictions are sensitive to background color.

Dataset layout expected on disk:
    <root>/colorized-MNIST-master/
        training/{digit}/{color}_{idx}.png
        testing/{digit}/{color}_{idx}.png

USAGE:
    from datasets import ColoredMNIST

    dataset = ColoredMNIST(root="./data", train=False)
    image, label, color_name, group_id = dataset[0]
    # image     : PIL Image (RGB), digit on a colored background
    # label     : int, true digit (0-9)
    # color_name: str, background color ("red", "green", or "blue")
    # group_id  : int = label * 3 + color_index
"""

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


# ── Color definitions ──────────────────────────────────────────────────────────

COLORS = ["red", "green", "blue"]
COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2}


class ColoredMNIST(Dataset):
    """
    Loads all images from the Colored MNIST dataset without any modification.

    Each sample returns:
        image      (PIL.Image RGB)  : digit on a colored background
        label      (int)            : true digit class (0-9)
        color_name (str)            : background color ("red", "green", or "blue")
        group_id   (int)            : label * 3 + color_index
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = False,
        download: bool = False,   # kept for call-site compatibility; data must be on disk
        clip_preprocess=None,
        max_samples: int = None,
        seed: int = 42,
        # Legacy parameter — ignored, kept so existing call sites don't break
        split: str = None,
    ):
        """
        Args:
            root          : parent directory containing 'colorized-MNIST-master/'
            train         : if True, load from training/; otherwise testing/
            clip_preprocess: CLIP preprocessing transform, injected by CLIPZeroShot
            max_samples   : cap total loaded samples (useful for quick testing)
            seed          : random seed for subsampling
        """
        _ = download  # data must already be on disk
        _ = split     # no longer used; all images are loaded

        self.clip_preprocess = clip_preprocess
        self.rng = np.random.default_rng(seed)

        subset = "training" if train else "testing"
        dataset_root = os.path.join(root, "colorized-MNIST-master", subset)

        if not os.path.isdir(dataset_root):
            raise FileNotFoundError(
                f"Colored MNIST not found at: {dataset_root}\n"
                f"Download from: https://www.kaggle.com/datasets/youssifhisham/colored-mnist-dataset\n"
                f"and place it so that '{os.path.join(root, 'colorized-MNIST-master')}' exists."
            )

        self.samples = self._load_samples(dataset_root, max_samples)

    def _load_samples(self, dataset_root: str, max_samples: int) -> list:
        """Walk digit subdirectories and collect all (path, label, color, group_id) tuples."""
        samples = []

        for digit in range(10):
            digit_dir = os.path.join(dataset_root, str(digit))
            if not os.path.isdir(digit_dir):
                continue

            for fname in sorted(os.listdir(digit_dir)):
                if not fname.endswith(".png"):
                    continue

                # Filename format: {color}_{original_mnist_index}.png
                color_name = fname.split("_")[0]
                if color_name not in COLOR_TO_IDX:
                    continue

                img_path  = os.path.join(digit_dir, fname)
                color_idx = COLOR_TO_IDX[color_name]
                group_id  = digit * 3 + color_idx
                samples.append((img_path, digit, color_name, group_id))

        if max_samples is not None and len(samples) > max_samples:
            chosen = self.rng.choice(len(samples), size=max_samples, replace=False)
            samples = [samples[i] for i in chosen]

        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label, color_name, group_id = self.samples[idx]
        pil_image = Image.open(img_path).convert("RGB")

        if self.clip_preprocess is not None:
            image = self.clip_preprocess(pil_image)
        else:
            image = pil_image

        return image, label, color_name, group_id

    def get_group_info(self) -> dict:
        """Returns {(label, color_name): count} for inspection."""
        from collections import Counter
        counts = Counter((label, color_name)
                         for _, label, color_name, _ in self.samples)
        return dict(sorted(counts.items()))

    def __repr__(self):
        return (f"ColoredMNIST("
                f"n_samples={len(self)}, n_classes=10, n_colors=3)")
