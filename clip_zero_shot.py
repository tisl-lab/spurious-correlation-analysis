"""
clip_zero_shot.py
=================

CLIP Zero-Shot Inference Engine
--------------------------------

This module is the heart of the pipeline. It handles:

1. LOADING CLIP (ViT-B/32 from OpenAI)
2. TEXT PROMPT ENGINEERING
   - Shape prompts: describe the class WITHOUT mentioning color
   - Color prompts: describe what color each class is associated with
   - Combined: run both and compare
3. ZERO-SHOT CLASSIFICATION
   - Encode all text prompts once → text feature matrix
   - For each image batch: encode image → compute cosine similarity → argmax
4. RETURNING STRUCTURED RESULTS
   - Returns predictions alongside true labels, colors, and group IDs
   - Everything needed for analysis.py to compute per-group accuracy

ZERO-SHOT HOW IT WORKS:
    1. We define N text descriptions (one per class).
    2. CLIP encodes each description → text feature vector.
    3. For each image, CLIP encodes it → image feature vector.
    4. Prediction = argmax of cosine similarity(image_feat, all_text_feats).
    5. No training needed! CLIP has seen enough of the world to match these.

WHY TEXT PROMPT DESIGN MATTERS FOR SPURIOUS CORRELATION:
    Shape prompts ("a photo of the digit zero") tell CLIP to use shape.
    Color prompts ("a photo of a red digit") tell CLIP to use color.
    By comparing which prompt strategy wins in aligned vs. misaligned groups,
    we reveal how strongly CLIP has internalized color as a shortcut.
"""

import torch
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

try:
    import clip
except ImportError:
    raise ImportError(
        "OpenAI CLIP is not installed. Install it with:\n"
        "  pip install git+https://github.com/openai/CLIP.git"
    )


# ── Text Prompt Templates ──────────────────────────────────────────────────────

# MNIST: shape-based prompts — describe digit shape, ignore color
MNIST_SHAPE_PROMPTS = {
    0: "a photo of the handwritten digit zero",
    1: "a photo of the handwritten digit one",
    2: "a photo of the handwritten digit two",
    3: "a photo of the handwritten digit three",
    4: "a photo of the handwritten digit four",
    5: "a photo of the handwritten digit five",
    6: "a photo of the handwritten digit six",
    7: "a photo of the handwritten digit seven",
    8: "a photo of the handwritten digit eight",
    9: "a photo of the handwritten digit nine",
}

# MNIST: color-based prompts — describe by the background color that is spuriously correlated.
# The Colored MNIST dataset uses red/green/blue backgrounds, cycling across digits 0-9.
# Aligned assignment: 0→red, 1→green, 2→blue, 3→red, 4→green, 5→blue, 6→red, 7→green, 8→blue, 9→red
MNIST_COLOR_PROMPTS = {
    0: "a photo of a handwritten digit on a red background",
    1: "a photo of a handwritten digit on a green background",
    2: "a photo of a handwritten digit on a blue background",
    3: "a photo of a handwritten digit on a red background",
    4: "a photo of a handwritten digit on a green background",
    5: "a photo of a handwritten digit on a blue background",
    6: "a photo of a handwritten digit on a red background",
    7: "a photo of a handwritten digit on a green background",
    8: "a photo of a handwritten digit on a blue background",
    9: "a photo of a handwritten digit on a red background",
}

# CIFAR-10: shape/semantic prompts
CIFAR10_SHAPE_PROMPTS = {
    0: "a photo of an airplane",
    1: "a photo of an automobile",
    2: "a photo of a bird",
    3: "a photo of a cat",
    4: "a photo of a deer",
    5: "a photo of a dog",
    6: "a photo of a frog",
    7: "a photo of a horse",
    8: "a photo of a ship",
    9: "a photo of a truck",
}

# CIFAR-10: context-based prompts — describe the background environment each class
# is spuriously correlated with in the aligned split.
# sky   → airplane, bird       (objects naturally in the sky)
# water → frog, ship           (objects naturally in/on water)
# land  → all others           (objects naturally on the ground)
CIFAR10_COLOR_PROMPTS = {
    0: "a photo of an object against a blue sky background",       # airplane → sky
    1: "a photo of a vehicle on a land or road background",        # automobile → land
    2: "a photo of an animal against a blue sky background",       # bird → sky
    3: "a photo of an animal on a land background",                # cat → land
    4: "a photo of an animal on a land or nature background",      # deer → land
    5: "a photo of an animal on a land background",                # dog → land
    6: "a photo of an animal near or in water",                    # frog → water
    7: "a photo of an animal on a land background",                # horse → land
    8: "a photo of a vessel on a water background",                # ship → water
    9: "a photo of a vehicle on a land or road background",        # truck → land
}

PROMPT_SETS = {
    "mnist": {
        "shape": MNIST_SHAPE_PROMPTS,
        "color": MNIST_COLOR_PROMPTS,
    },
    "cifar10": {
        "shape": CIFAR10_SHAPE_PROMPTS,
        "color": CIFAR10_COLOR_PROMPTS,
    }
}


# ── CLIP Model Wrapper ─────────────────────────────────────────────────────────

class CLIPZeroShot:
    """
    A clean wrapper around OpenAI's CLIP for zero-shot classification.

    Usage:
        model = CLIPZeroShot(device="cuda")
        results = model.run(dataset, prompt_mode="shape", dataset_name="mnist")
    """

    def __init__(self, model_name: str = "ViT-B/32", device: str = None):
        """
        Args:
            model_name : CLIP model variant. ViT-B/32 is the standard baseline.
                         Other options: "RN50", "RN101", "ViT-L/14"
            device     : "cuda", "cpu", or None (auto-detect)
        """
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        self.device = device
        self.model_name = model_name

        print(f"Loading CLIP {model_name} on {device}...")
        self.model, self.preprocess = clip.load(model_name, device=device)
        self.model.eval()
        print(f"CLIP loaded. Parameters: {sum(p.numel() for p in self.model.parameters()):,}")

    def encode_text_prompts(self, prompts: dict) -> torch.Tensor:
        """
        Encode a dict of {class_id: text_prompt} into a normalized feature matrix.

        Returns:
            torch.Tensor of shape (n_classes, embed_dim), L2-normalized
        """
        n_classes = len(prompts)
        texts = [prompts[i] for i in range(n_classes)]

        print(f"  Encoding {n_classes} text prompts...")
        with torch.no_grad():
            tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(tokens)
            text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        return text_features  # (n_classes, D)

    def fine_tune(
        self,
        dataset,
        dataset_name: str,
        epochs: int = 3,
        lr: float = 1e-5,
        batch_size: int = 64,
    ):
        """
        Fine-tune the CLIP image encoder on a labeled dataset.

        Only the image encoder is updated; the text encoder stays frozen.
        Classification logits are computed the same way as zero-shot inference
        (cosine similarity × 100), so the fine-tuned model can still be
        evaluated with run() afterwards.

        Args:
            dataset      : ColoredMNIST instance (train split)
            dataset_name : "mnist" or "cifar10"
            epochs       : number of passes over the training data
            lr           : learning rate for Adam (keep small, e.g. 1e-5)
            batch_size   : training batch size
        """
        import torch.nn.functional as F

        dataset.clip_preprocess = self.preprocess
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        # Pre-compute text features NOW, while the model and tokenizer are still
        # on self.device. We move the model to CPU right after (for MPS stability),
        # so text features must be encoded before that move.
        shape_prompts = PROMPT_SETS[dataset_name]["shape"]
        with torch.no_grad():
            text_features = self.encode_text_prompts(shape_prompts).float()  # (C, D)

        # MPS has known autograd bugs with transformer backpropagation that produce
        # NaN gradients and crash on the second epoch. Fine-tune on CPU instead;
        # the model is moved back to MPS afterwards for fast inference.
        train_device = "cpu" if self.device == "mps" else self.device
        if train_device != self.device:
            print(f"  ⚠ MPS detected — fine-tuning on CPU for numerical stability "
                  f"(model will return to MPS for inference)")
            self.model.to(train_device)
            text_features = text_features.to(train_device)

        # Freeze text encoder, unfreeze image encoder only
        for param in self.model.parameters():
            param.requires_grad = False
        for param in self.model.visual.parameters():
            param.requires_grad = True

        # CLIP's ViT-B/32 has mixed dtypes: 98 float16 + 54 float32 params in the
        # visual encoder. Converting only visual to float32 changes self.dtype
        # (CLIP's dtype property reads visual.conv1.weight.dtype), which causes
        # encode_text to cast inputs to float32 but run through float16 text-encoder
        # weights → MPS matmul dtype crash. Convert the full model to float32.
        self.model.float()

        optimizer = torch.optim.Adam(self.model.visual.parameters(), lr=lr)

        logit_scale = self.model.logit_scale.exp().float().item()

        print(f"  Fine-tuning image encoder for {epochs} epoch(s)  "
              f"(lr={lr}, batch={batch_size}, n={len(dataset)} images, "
              f"device={train_device})")

        self.model.train()
        for epoch in range(epochs):
            total_loss, n_batches = 0.0, 0
            for images, labels, _, _ in tqdm(loader, desc=f"  Epoch {epoch+1}/{epochs}"):
                images = images.to(train_device).float()
                labels = labels.to(train_device)

                image_features = self.model.encode_image(images).float()
                # clamp prevents zero-norm → NaN; avoid Python float literals that
                # can silently upcast tensor dtype on MPS/CUDA
                image_features = image_features / image_features.norm(dim=-1, keepdim=True).clamp(min=1e-8)

                logits = logit_scale * image_features @ text_features.T  # (B, C)
                loss   = F.cross_entropy(logits, labels)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.visual.parameters(), max_norm=1.0)
                optimizer.step()

                total_loss += loss.item()
                n_batches  += 1

            print(f"    avg loss: {total_loss / n_batches:.4f}")

        self.model.eval()

        # Move model back to the original device for inference
        if train_device != self.device:
            self.model.to(self.device)

        # Re-enable gradients for all parameters (clean state for any future use)
        for param in self.model.parameters():
            param.requires_grad = True

    @torch.no_grad()
    def run(
        self,
        dataset,
        prompt_mode: str,
        dataset_name: str,
        batch_size: int = 64,
    ) -> dict:
        """
        Run zero-shot inference on a dataset and return structured results.

        Args:
            dataset      : ColoredMNIST or ColoredCIFAR10 instance
                           (will have clip_preprocess injected automatically)
            prompt_mode  : "shape", "color", or "both"
            dataset_name : "mnist" or "cifar10"
            batch_size   : inference batch size

        Returns:
            dict with keys:
                "predictions_shape" : np.array (N,) — predicted class indices using shape prompts
                "predictions_color" : np.array (N,) — predicted class indices using color prompts
                "true_labels"       : np.array (N,) — ground truth labels
                "color_names"       : list of str   — color/tint applied to each sample
                "group_ids"         : np.array (N,) — (label * 10 + color_idx)
                "prompt_mode"       : str
                "dataset_name"      : str
                "shape_prompts"     : dict (for logging)
                "color_prompts"     : dict (for logging)
        """
        assert prompt_mode in ("shape", "color", "both"), \
            f"prompt_mode must be 'shape', 'color', or 'both'. Got: {prompt_mode}"
        assert dataset_name in PROMPT_SETS, \
            f"dataset_name must be one of {list(PROMPT_SETS.keys())}. Got: {dataset_name}"

        # Inject CLIP preprocess into dataset
        dataset.clip_preprocess = self.preprocess

        # Build DataLoader
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,  # 0 is safer for PIL images
            collate_fn=self._collate_fn,
        )

        # Encode only the prompt types that are needed
        ### shape prompts should be : 0 ,1 , 2  , 3, 4 ,5 , 6, 7, 8, 9
        ### color prompts should be : red , blue and green
        
        run_shape = prompt_mode in ("shape", "both")
        run_color = prompt_mode in ("color", "both")

        shape_prompts = PROMPT_SETS[dataset_name]["shape"]
        color_prompts = PROMPT_SETS[dataset_name]["color"]

        # Cast to float32 to ensure consistency with image features.
        # After fine_tune(), the visual encoder is float32 while the text encoder
        # may still be float16; mixing dtypes crashes MPS matrix multiply.
        text_feats_shape = self.encode_text_prompts(shape_prompts).float() if run_shape else None
        text_feats_color = self.encode_text_prompts(color_prompts).float() if run_color else None

        # Run inference
        all_preds_shape = [] if run_shape else None
        all_preds_color = [] if run_color else None
        all_labels      = []
        all_color_names = []
        all_group_ids   = []

        print(f"  Running inference on {len(dataset)} samples...")
        for images, labels, color_names, group_ids in tqdm(loader, desc="Inference"):
            images = images.to(self.device)

            # Encode images → L2-normalized feature vector (float32 for MPS dtype safety)
            image_features = self.model.encode_image(images).float()
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)

            # Cosine similarity (scaled by 100 to match CLIP's convention)
            if run_shape:
                logits = 100.0 * image_features @ text_feats_shape.T  # (B, C)
                all_preds_shape.extend(logits.argmax(dim=-1).cpu().numpy())

            if run_color:
                logits = 100.0 * image_features @ text_feats_color.T  # (B, C)
                all_preds_color.extend(logits.argmax(dim=-1).cpu().numpy())

            all_labels.extend(labels.numpy())
            all_color_names.extend(color_names)
            all_group_ids.extend(group_ids.numpy())

        results = {
            "true_labels":   np.array(all_labels),
            "color_names":   all_color_names,
            "group_ids":     np.array(all_group_ids),
            "prompt_mode":   prompt_mode,
            "dataset_name":  dataset_name,
        }
        if run_shape:
            results["predictions_shape"] = np.array(all_preds_shape)
            results["shape_prompts"]     = shape_prompts
        if run_color:
            results["predictions_color"] = np.array(all_preds_color)
            results["color_prompts"]     = color_prompts

        return results

    @staticmethod
    def _collate_fn(batch):
        """Custom collate to handle mixed types (tensors + strings + ints)."""
        images     = torch.stack([item[0] for item in batch])
        labels     = torch.tensor([item[1] for item in batch], dtype=torch.long)
        color_names = [item[2] for item in batch]
        group_ids  = torch.tensor([item[3] for item in batch], dtype=torch.long)
        return images, labels, color_names, group_ids
