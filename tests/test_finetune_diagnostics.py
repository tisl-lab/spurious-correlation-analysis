"""
tests/test_finetune_diagnostics.py
===================================
Diagnostic tests to identify why fine-tuning collapses all predictions to class 0.

Run standalone:
    python tests/test_finetune_diagnostics.py

Or via pytest:
    pytest tests/test_finetune_diagnostics.py -v -s
"""

import copy
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from clip_zero_shot import CLIPZeroShot
from datasets import ColoredMNIST

# ── helpers ───────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
N_SAMPLES = 120   # keep small so tests run fast


def _get_device():
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _load_small_dataset(n=N_SAMPLES, seed=0):
    return ColoredMNIST(root=DATA_DIR, train=False, max_samples=n, seed=seed)


def _split(dataset, train_pct=0.5, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(dataset.samples))
    n_train = max(1, int(len(dataset.samples) * train_pct))
    train_ds = copy.copy(dataset)
    test_ds  = copy.copy(dataset)
    train_ds.samples = [dataset.samples[i] for i in idx[:n_train]]
    test_ds.samples  = [dataset.samples[i] for i in idx[n_train:]]
    return train_ds, test_ds


def _get_image_features(model_wrapper, dataset, n=20):
    """Return a small batch of L2-normalised image features."""
    from torch.utils.data import DataLoader
    dataset.clip_preprocess = model_wrapper.preprocess
    loader = DataLoader(dataset, batch_size=n, shuffle=False,
                        collate_fn=model_wrapper._collate_fn)
    images, *_ = next(iter(loader))
    with torch.no_grad():
        feats = model_wrapper.model.encode_image(images.to(model_wrapper.device)).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
    return feats  # (n, D)


# ── tests ─────────────────────────────────────────────────────────────────────

def test_model_dtype():
    """CLIP loads in float16 on GPU/MPS — confirm and warn."""
    device = _get_device()
    m = CLIPZeroShot(model_name="ViT-B/32", device=device)
    dtype = next(m.model.visual.parameters()).dtype
    print(f"\n[dtype]  visual encoder dtype = {dtype}  (device={device})")
    if dtype == torch.float16:
        print("  ⚠  float16 detected — Adam optimizer is numerically unstable at fp16")
        print("     gradients can underflow → parameters become NaN → argmax always 0")
    else:
        print("  ✓  float32 — dtype is fine")
    # not a hard assertion; this is diagnostic
    return dtype


def test_predictions_before_finetune():
    """Zero-shot predictions must span more than 1 class."""
    device = _get_device()
    m = CLIPZeroShot(model_name="ViT-B/32", device=device)
    ds = _load_small_dataset()
    results = m.run(dataset=ds, prompt_mode="shape",
                    dataset_name="mnist", batch_size=64)
    preds = results["predictions_shape"]
    n_unique = len(np.unique(preds))
    dist = np.bincount(preds, minlength=10)
    print(f"\n[pre-FT]  unique classes predicted: {n_unique}/10")
    print(f"          distribution: {dist}")
    assert n_unique > 1, f"FAIL: all {len(preds)} predictions = {preds[0]} even before fine-tuning"
    print("          ✓ predictions are diverse before fine-tuning")


def test_features_not_nan_after_finetune():
    """Image features must not be NaN or Inf after fine-tuning."""
    device = _get_device()
    m = CLIPZeroShot(model_name="ViT-B/32", device=device)
    ds = _load_small_dataset()
    train_ds, test_ds = _split(ds, train_pct=0.6)

    feats_before = _get_image_features(m, test_ds)
    print(f"\n[NaN check]  features BEFORE FT — NaN:{torch.isnan(feats_before).any().item()}  "
          f"Inf:{torch.isinf(feats_before).any().item()}")

    m.fine_tune(dataset=train_ds, dataset_name="mnist",
                epochs=2, lr=1e-5, batch_size=32)

    feats_after = _get_image_features(m, test_ds)
    has_nan = torch.isnan(feats_after).any().item()
    has_inf = torch.isinf(feats_after).any().item()
    print(f"             features AFTER  FT — NaN:{has_nan}  Inf:{has_inf}")
    if has_nan or has_inf:
        print("  ✗ FOUND THE BUG: features are corrupted after fine-tuning!")
        print("    → float16 Adam instability: gradients underflow/overflow → NaN params")
    assert not has_nan, "Image features contain NaN after fine-tuning (float16 training instability)"
    assert not has_inf, "Image features contain Inf after fine-tuning"
    print("  ✓ features are clean after fine-tuning")


def test_loss_decreases():
    """Training loss should decrease across epochs (basic sanity)."""
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from clip_zero_shot import PROMPT_SETS

    device = _get_device()
    m = CLIPZeroShot(model_name="ViT-B/32", device=device)
    ds = _load_small_dataset(n=60)

    ds.clip_preprocess = m.preprocess
    loader = DataLoader(ds, batch_size=30, shuffle=True,
                        collate_fn=m._collate_fn)

    for p in m.model.parameters():
        p.requires_grad = False
    for p in m.model.visual.parameters():
        p.requires_grad = True

    m.model.visual.float()  # cast to fp32

    opt = torch.optim.Adam(m.model.visual.parameters(), lr=1e-5)
    text_feats = m.encode_text_prompts(PROMPT_SETS["mnist"]["shape"]).float()

    losses = []
    m.model.train()
    for _ in range(3):
        batch_losses = []
        for imgs, labels, _, _ in loader:
            imgs   = imgs.to(device)
            labels = labels.to(device)
            feats  = m.model.encode_image(imgs).float()
            feats  = feats / feats.norm(dim=-1, keepdim=True)
            loss   = F.cross_entropy(100.0 * feats @ text_feats.T, labels)
            opt.zero_grad()
            loss.backward()
            opt.step()
            batch_losses.append(loss.item())
        losses.append(np.mean(batch_losses))
    m.model.eval()
    for p in m.model.parameters():
        p.requires_grad = True

    print(f"\n[loss]  per-epoch losses: {[f'{l:.4f}' for l in losses]}")
    valid = all(not np.isnan(l) and not np.isinf(l) for l in losses)
    assert valid, f"Loss is NaN/Inf: {losses}"
    print("  ✓ loss is finite across all epochs")
    if losses[-1] < losses[0]:
        print("  ✓ loss decreased (training is learning)")
    else:
        print("  ⚠  loss did not decrease — model may not be learning effectively")


def test_predictions_after_finetune():
    """Predictions must remain diverse after fine-tuning."""
    device = _get_device()
    m = CLIPZeroShot(model_name="ViT-B/32", device=device)
    ds = _load_small_dataset()
    train_ds, test_ds = _split(ds, train_pct=0.6)

    m.fine_tune(dataset=train_ds, dataset_name="mnist",
                epochs=2, lr=1e-5, batch_size=32)

    results = m.run(dataset=test_ds, prompt_mode="shape",
                    dataset_name="mnist", batch_size=64)
    preds = results["predictions_shape"]
    n_unique = len(np.unique(preds))
    dist = np.bincount(preds, minlength=10)
    print(f"\n[post-FT]  unique classes predicted: {n_unique}/10")
    print(f"           distribution: {dist}")
    assert n_unique > 1, (
        f"FAIL: all {len(preds)} predictions = {preds[0]} after fine-tuning\n"
        f"      This is the collapse-to-class-0 bug."
    )
    print("  ✓ predictions are diverse after fine-tuning")


# ── standalone runner ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  FINE-TUNING DIAGNOSTICS")
    print("=" * 60)

    results = {}

    tests = [
        ("dtype check",              test_model_dtype),
        ("pre-FT predictions",       test_predictions_before_finetune),
        ("NaN in features after FT", test_features_not_nan_after_finetune),
        ("loss decreases",           test_loss_decreases),
        ("post-FT predictions",      test_predictions_after_finetune),
    ]

    for name, fn in tests:
        print(f"\n{'─'*60}\nRunning: {name}")
        try:
            fn()
            results[name] = "PASS"
        except AssertionError as e:
            print(f"  ✗ FAIL: {e}")
            results[name] = "FAIL"
        except Exception as e:
            print(f"  ✗ ERROR: {e}")
            results[name] = f"ERROR: {e}"
        finally:
            # Release MPS memory between tests to prevent state pollution
            # between multiple model instances within the same process.
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()

    print(f"\n{'='*60}")
    print("  SUMMARY")
    print(f"{'='*60}")
    for name, status in results.items():
        icon = "✓" if status == "PASS" else "✗"
        print(f"  {icon}  {name:<35} {status}")
    print()
