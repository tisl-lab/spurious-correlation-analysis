"""
tests/test_msae_ftclip.py
=========================
Unit tests for msae_ftclip.py functions.

Covers:
  1. make_projection_ablation_hook  — direction zeroing math
  2. make_sae_ablation_hook         — encode→zero→decode round-trip
  3. save_representations           — file creation and content
  4. Corrected/harmed image logic   — logit-delta scoring
  5. ManifestDataset                — CSV loading and sample structure

All tests run without real CLIP or SAE weights (mock objects used).

Run with:
    python -m pytest tests/test_msae_ftclip.py -v
"""

import os
import sys
import tempfile
import csv

import numpy as np
import torch
import pandas as pd
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from msae_ftclip import (
    make_projection_ablation_hook,
    make_sae_ablation_hook,
    save_representations,
    ManifestDataset,
)


# ─────────────────────────────────────────────────────────────────────────────
# 1. make_projection_ablation_hook
# ─────────────────────────────────────────────────────────────────────────────

def test_projection_hook_zeros_target_directions():
    """After the hook, each concept direction should have zero component in output."""
    D = 8
    # Two orthogonal unit dirs: e0 and e1
    dirs = torch.eye(D)[:2]                          # shape (2, D)
    hook_fn = make_projection_ablation_hook(dirs)

    feat = torch.ones(1, D)                          # all-ones vector
    result = hook_fn(module=None, input=None, output=feat)

    assert result.shape == feat.shape
    # Component along e0 and e1 must be ~0
    assert result[0, 0].abs().item() < 1e-5, "e0 component not zeroed"
    assert result[0, 1].abs().item() < 1e-5, "e1 component not zeroed"
    # Other dims untouched
    for i in range(2, D):
        assert abs(result[0, i].item() - 1.0) < 1e-5, f"dim {i} should be unchanged"


def test_projection_hook_batch():
    """Hook must handle batches correctly — each row processed independently."""
    D = 4
    dirs = torch.eye(D)[:1]                          # ablate only dim 0
    hook_fn = make_projection_ablation_hook(dirs)

    feat = torch.tensor([[3.0, 1.0, 2.0, 5.0],
                          [7.0, 0.0, 4.0, 1.0]])    # (2, 4)
    result = hook_fn(None, None, feat)

    assert result.shape == (2, 4)
    assert result[0, 0].abs().item() < 1e-5
    assert result[1, 0].abs().item() < 1e-5
    # Other dims unchanged
    assert abs(result[0, 1].item() - 1.0) < 1e-5
    assert abs(result[1, 2].item() - 4.0) < 1e-5


def test_projection_hook_preserves_dtype():
    """Output dtype must match the input tensor dtype."""
    dirs = torch.eye(4)[:1]
    hook_fn = make_projection_ablation_hook(dirs)

    feat_fp16 = torch.ones(1, 4, dtype=torch.float16)
    result = hook_fn(None, None, feat_fp16)
    assert result.dtype == torch.float16, "dtype not preserved"


def test_projection_hook_no_dirs():
    """With an empty direction list the output should equal the input."""
    dirs = torch.zeros(0, 4)
    hook_fn = make_projection_ablation_hook(dirs)

    feat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = hook_fn(None, None, feat)
    torch.testing.assert_close(result, feat)


# ─────────────────────────────────────────────────────────────────────────────
# 2. make_sae_ablation_hook
# ─────────────────────────────────────────────────────────────────────────────

class _MockSAE:
    """Minimal SAE stand-in: identity encoder/decoder with known latent values."""

    def __init__(self, D=6, L=10):
        self.D = D
        self.L = L

    def encode(self, x):
        # latents[i] = x[i] padded with zeros to length L
        B = x.shape[0]
        latents = torch.zeros(B, self.L, dtype=x.dtype)
        latents[:, :self.D] = x
        return latents, latents.clone()

    def decode(self, latents):
        # reconstruct by taking first D dims and multiplying by 2
        return latents[:, :self.D] * 2.0


def test_sae_hook_zeros_specified_latents():
    """Zeroing concept_idx latents must produce a different (ablated) output."""
    D, L = 6, 10
    sae = _MockSAE(D, L)
    concept_idx = [0, 2]                             # ablate dims 0 and 2

    hook_fn = make_sae_ablation_hook(sae, concept_idx, device="cpu")
    feat = torch.ones(1, D)
    result = hook_fn(None, None, feat)

    assert result.shape == (1, D)
    # dims 0 and 2 were zeroed in latents before decode, so result[0,0] and [0,2] == 0
    assert result[0, 0].abs().item() < 1e-5, "latent 0 should have been zeroed"
    assert result[0, 2].abs().item() < 1e-5, "latent 2 should have been zeroed"
    # non-ablated dims should be 1*2 = 2
    for i in [1, 3, 4, 5]:
        assert abs(result[0, i].item() - 2.0) < 1e-5, f"dim {i} should be 2.0"


def test_sae_hook_no_concept_idx():
    """With an empty concept list the hook returns the decoded reconstruction unchanged."""
    sae = _MockSAE(D=4, L=8)
    hook_fn = make_sae_ablation_hook(sae, [], device="cpu")

    feat = torch.tensor([[1.0, 2.0, 3.0, 4.0]])
    result = hook_fn(None, None, feat)
    # decode multiplies by 2 — no concepts zeroed
    expected = torch.tensor([[2.0, 4.0, 6.0, 8.0]])
    torch.testing.assert_close(result, expected)


def test_sae_hook_preserves_dtype():
    """Output dtype must match input dtype."""
    sae = _MockSAE(D=4, L=8)
    hook_fn = make_sae_ablation_hook(sae, [0], device="cpu")

    feat = torch.ones(1, 4, dtype=torch.float16)
    result = hook_fn(None, None, feat)
    assert result.dtype == torch.float16


# ─────────────────────────────────────────────────────────────────────────────
# 3. save_representations
# ─────────────────────────────────────────────────────────────────────────────

def _make_dummy_results(N=10, D=4):
    paths = [f"/fake/img_{i:03d}.jpg" for i in range(N)]
    return {
        "clip_representations": torch.randn(N, D),
        "sae_representations":  torch.randn(N, D * 2),
        "sae_reconstructed":    torch.randn(N, D),
        "image_paths":          paths,
        "metrics": [
            {"fvu": float(i), "mae": float(i) * 0.1, "cs": 0.9, "l0": 5.0,
             "highest_magnitude": float(i)}
            for i in range(N)
        ],
    }


def test_save_representations_creates_files(tmp_path):
    results = _make_dummy_results()
    out_dir = save_representations(results, "test", str(tmp_path))

    assert os.path.isdir(out_dir)
    for fname in ["clip_representations.pt", "sae_representations.pt",
                  "sae_reconstructed.pt", "metrics.csv"]:
        assert os.path.isfile(os.path.join(out_dir, fname)), f"Missing: {fname}"


def test_save_representations_tensor_shapes(tmp_path):
    results = _make_dummy_results(N=8, D=6)
    out_dir = save_representations(results, "ft_train", str(tmp_path))

    clip = torch.load(os.path.join(out_dir, "clip_representations.pt"),
                      weights_only=False)
    sae  = torch.load(os.path.join(out_dir, "sae_representations.pt"),
                      weights_only=False)
    assert clip.shape == (8, 6)
    assert sae.shape  == (8, 12)


def test_save_representations_metrics_csv(tmp_path):
    N = 5
    results = _make_dummy_results(N=N)
    out_dir = save_representations(results, "test", str(tmp_path))

    df = pd.read_csv(os.path.join(out_dir, "metrics.csv"), index_col="img_path")
    assert len(df) == N
    assert "fvu" in df.columns
    assert df.index.tolist() == results["image_paths"]


def test_save_then_reload_roundtrip(tmp_path):
    """Tensors saved by save_representations must reload bit-exactly."""
    results = _make_dummy_results(N=6, D=3)
    out_dir = save_representations(results, "test", str(tmp_path))

    loaded_clip = torch.load(os.path.join(out_dir, "clip_representations.pt"),
                             weights_only=False)
    torch.testing.assert_close(loaded_clip, results["clip_representations"])


# ─────────────────────────────────────────────────────────────────────────────
# 4. Corrected / harmed image scoring logic
# ─────────────────────────────────────────────────────────────────────────────

def _ablation_corrected(orig_logits, ablated_logits, orig_preds, ablated_preds,
                        true_labels, group_ids, gid):
    """Replicate the corrected-image selection from ablate_spurious_concepts."""
    g_indices    = np.where(group_ids == gid)[0]
    was_wrong    = orig_preds[g_indices]    != true_labels[g_indices]
    now_correct  = ablated_preds[g_indices] == true_labels[g_indices]
    corrected    = g_indices[was_wrong & now_correct]
    if len(corrected) == 0:
        return corrected, np.array([])
    true_cls_idx = true_labels[corrected]
    scores = (ablated_logits[corrected, true_cls_idx] -
              orig_logits[corrected, true_cls_idx])
    top_k     = min(10, len(corrected))
    top_order = np.argsort(scores)[::-1][:top_k]
    return corrected[top_order], scores[top_order]


def _ablation_harmed(orig_logits, ablated_logits, orig_preds, ablated_preds,
                     true_labels, group_ids, gid):
    """Replicate the harmed-image selection from ablate_spurious_concepts."""
    g_indices   = np.where(group_ids == gid)[0]
    was_correct = orig_preds[g_indices]    == true_labels[g_indices]
    now_wrong   = ablated_preds[g_indices] != true_labels[g_indices]
    harmed      = g_indices[was_correct & now_wrong]
    if len(harmed) == 0:
        return harmed, np.array([])
    true_cls_idx = true_labels[harmed]
    scores = (orig_logits[harmed, true_cls_idx] -
              ablated_logits[harmed, true_cls_idx])
    top_k     = min(10, len(harmed))
    top_order = np.argsort(scores)[::-1][:top_k]
    return harmed[top_order], scores[top_order]


def test_corrected_images_identified_correctly():
    N = 6
    true_labels   = np.array([0, 1, 0, 1, 0, 1])
    group_ids     = np.array([1, 1, 1, 2, 2, 2])
    orig_preds    = np.array([1, 1, 1, 1, 0, 1])  # 0,2 wrong in grp1; 3 wrong in grp2
    ablated_preds = np.array([0, 1, 0, 1, 0, 1])  # 0,2 fixed; 3 fixed

    # Logits: ablated raises correct-class logit by 1.0 for corrected images
    orig_logits    = np.zeros((N, 2))
    ablated_logits = np.zeros((N, 2))
    ablated_logits[0, 0] = 2.0   # big gain
    ablated_logits[2, 0] = 1.0   # smaller gain
    ablated_logits[3, 1] = 1.5

    top_idx, scores = _ablation_corrected(
        orig_logits, ablated_logits, orig_preds, ablated_preds,
        true_labels, group_ids, gid=1
    )
    # Images 0 and 2 were corrected in group 1; image 0 has higher score
    assert set(top_idx.tolist()) == {0, 2}
    assert top_idx[0] == 0, "Image 0 has the biggest gain — should rank first"
    assert (scores >= 0).all(), "All corrected scores should be non-negative"


def test_harmed_images_identified_correctly():
    N = 6
    true_labels   = np.array([0, 1, 0, 1, 0, 1])
    group_ids     = np.array([0, 0, 3, 3, 3, 0])
    orig_preds    = np.array([0, 1, 0, 1, 0, 1])   # all correct
    ablated_preds = np.array([1, 1, 0, 0, 1, 1])   # 0,4 now wrong

    orig_logits    = np.ones((N, 2))
    ablated_logits = np.ones((N, 2))
    ablated_logits[0, 0] = -1.0   # big drop
    ablated_logits[4, 0] = 0.2    # smaller drop

    top_idx, scores = _ablation_harmed(
        orig_logits, ablated_logits, orig_preds, ablated_preds,
        true_labels, group_ids, gid=0
    )
    assert set(top_idx.tolist()) == {0}, "Only image 0 is in group 0 and harmed"
    assert scores[0] > 0, "Drop score should be positive"


def test_corrected_empty_when_none_fixed():
    N = 4
    true_labels   = np.array([0, 1, 0, 1])
    group_ids     = np.array([1, 1, 1, 1])
    orig_preds    = np.array([0, 1, 0, 1])   # all correct already
    ablated_preds = np.array([0, 1, 0, 1])

    orig_logits    = np.zeros((N, 2))
    ablated_logits = np.zeros((N, 2))

    top_idx, scores = _ablation_corrected(
        orig_logits, ablated_logits, orig_preds, ablated_preds,
        true_labels, group_ids, gid=1
    )
    assert len(top_idx) == 0


def test_corrected_capped_at_10():
    """Top-k is capped at 10 even when more images are corrected."""
    N = 20
    true_labels   = np.zeros(N, dtype=int)
    group_ids     = np.ones(N, dtype=int)
    orig_preds    = np.ones(N, dtype=int)    # all wrong
    ablated_preds = np.zeros(N, dtype=int)   # all fixed

    orig_logits    = np.zeros((N, 2))
    ablated_logits = np.random.rand(N, 2)

    top_idx, scores = _ablation_corrected(
        orig_logits, ablated_logits, orig_preds, ablated_preds,
        true_labels, group_ids, gid=1
    )
    assert len(top_idx) == 10


# ─────────────────────────────────────────────────────────────────────────────
# 5. ManifestDataset
# ─────────────────────────────────────────────────────────────────────────────

def _write_manifest(path, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["img_path", "label", "bg", "group_id"])
        writer.writeheader()
        writer.writerows(rows)


def test_manifest_dataset_len_and_samples(tmp_path):
    rows = [
        {"img_path": "/a/1.jpg", "label": 0, "bg": "land", "group_id": 0},
        {"img_path": "/a/2.jpg", "label": 1, "bg": "water", "group_id": 3},
        {"img_path": "/a/3.jpg", "label": 0, "bg": "water", "group_id": 1},
    ]
    csv_path = str(tmp_path / "manifest.csv")
    _write_manifest(csv_path, rows)

    ds = ManifestDataset(csv_path)
    assert len(ds) == 3
    assert ds.samples[0] == ("/a/1.jpg", 0, "land", 0)
    assert ds.samples[1] == ("/a/2.jpg", 1, "water", 3)
    assert ds.samples[2] == ("/a/3.jpg", 0, "water", 1)


def test_manifest_dataset_types(tmp_path):
    rows = [{"img_path": "/x.jpg", "label": "1", "bg": "sky", "group_id": "2"}]
    csv_path = str(tmp_path / "manifest.csv")
    _write_manifest(csv_path, rows)

    ds = ManifestDataset(csv_path)
    path, label, bg, gid = ds.samples[0]
    assert isinstance(label, int), "label must be cast to int"
    assert isinstance(gid,   int), "group_id must be cast to int"
    assert isinstance(bg,    str)
    assert isinstance(path,  str)


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import tempfile, pathlib

    tests = [
        # projection hook
        test_projection_hook_zeros_target_directions,
        test_projection_hook_batch,
        test_projection_hook_preserves_dtype,
        test_projection_hook_no_dirs,
        # SAE hook
        test_sae_hook_zeros_specified_latents,
        test_sae_hook_no_concept_idx,
        test_sae_hook_preserves_dtype,
        # save_representations
        lambda: test_save_representations_creates_files(pathlib.Path(tempfile.mkdtemp())),
        lambda: test_save_representations_tensor_shapes(pathlib.Path(tempfile.mkdtemp())),
        lambda: test_save_representations_metrics_csv(pathlib.Path(tempfile.mkdtemp())),
        lambda: test_save_then_reload_roundtrip(pathlib.Path(tempfile.mkdtemp())),
        # corrected/harmed logic
        test_corrected_images_identified_correctly,
        test_harmed_images_identified_correctly,
        test_corrected_empty_when_none_fixed,
        test_corrected_capped_at_10,
        # ManifestDataset
        lambda: test_manifest_dataset_len_and_samples(pathlib.Path(tempfile.mkdtemp())),
        lambda: test_manifest_dataset_types(pathlib.Path(tempfile.mkdtemp())),
    ]

    passed = failed = 0
    print("\n" + "=" * 60)
    print("  msae_ftclip — unit tests")
    print("=" * 60)
    for fn in tests:
        name = getattr(fn, "__name__", "<lambda>")
        try:
            fn()
            print(f"  ✓ {name}")
            passed += 1
        except Exception as e:
            print(f"  ✗ {name}: {e}")
            failed += 1
    print(f"\n  {passed} passed, {failed} failed")
    print("=" * 60)
    sys.exit(0 if failed == 0 else 1)
