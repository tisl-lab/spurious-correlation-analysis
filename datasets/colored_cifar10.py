"""
datasets/colored_cifar10.py
===========================

Standard CIFAR-10 Dataset Loader (no modification)
---------------------------------------------------

Loads CIFAR-10 images exactly as they are — no tinting, no background replacement.
The analysis will measure CLIP's zero-shot per-class accuracy on unmodified images.

USAGE:
    from datasets import ColoredCIFAR10

    dataset = ColoredCIFAR10(root="./data", download=True)
    image, label, class_name, group_id = dataset[0]
    # image      : PIL Image (RGB), original CIFAR-10 image
    # label      : int, true class (0-9)
    # class_name : str, class label (e.g. "airplane")
    # group_id   : int = label  (one group per class)
"""

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import datasets


CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]


class ColoredCIFAR10(Dataset):
    """
    Standard CIFAR-10 dataset with no image modification.

    The class name is returned as the 'context' field so this dataset
    is compatible with the same analysis pipeline as ColoredMNIST.
    Since every image's context equals its class name, the spurious-gap
    metric will report N/A — correctly indicating no spurious color
    variation exists in unmodified CIFAR-10.

    Each sample returns:
        image      (PIL.Image RGB)  : original CIFAR-10 image
        label      (int)            : true class index (0-9)
        class_name (str)            : class label (e.g. "airplane")
        group_id   (int)            : equal to label
    """

    def __init__(
        self,
        root: str = "./data",
        train: bool = False,
        download: bool = True,
        clip_preprocess=None,
        max_samples: int = None,
        seed: int = 42,
        # Legacy parameters — ignored, kept so existing call sites don't break
        split: str = None,
        tint_alpha: float = None,
    ):
        """
        Args:
            root          : directory to download/cache CIFAR-10
            train         : if True, use training split
            download      : auto-download CIFAR-10 if not present
            clip_preprocess: CLIP preprocessing transform, injected by CLIPZeroShot
            max_samples   : cap total samples (for quick testing)
            seed          : random seed for subsampling
        """
        _ = split      # no longer used
        _ = tint_alpha # no longer used

        self.clip_preprocess = clip_preprocess
        self.rng = np.random.default_rng(seed)

        cifar = datasets.CIFAR10(root=root, train=train, download=download)
        images = cifar.data                # (N, 32, 32, 3), uint8
        labels = np.array(cifar.targets)   # (N,)

        if max_samples is not None:
            idx = self.rng.choice(len(labels), size=min(max_samples, len(labels)), replace=False)
            images, labels = images[idx], labels[idx]

        self.samples = [
            (Image.fromarray(img, mode="RGB"), int(lbl), CIFAR10_CLASSES[int(lbl)], int(lbl))
            for img, lbl in zip(images, labels)
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pil_image, label, class_name, group_id = self.samples[idx]

        if self.clip_preprocess is not None:
            image = self.clip_preprocess(pil_image)
        else:
            image = pil_image

        return image, label, class_name, group_id

    def get_group_info(self) -> dict:
        from collections import Counter
        counts = Counter((label, cls) for _, label, cls, _ in self.samples)
        return dict(sorted(counts.items()))

    def get_class_name(self, label: int) -> str:
        return CIFAR10_CLASSES[label]

    def __repr__(self):
        return (f"ColoredCIFAR10("
                f"n_samples={len(self)}, n_classes=10)")
