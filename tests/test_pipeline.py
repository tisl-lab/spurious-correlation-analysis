"""
tests/test_pipeline.py
======================

Sanity checks for the CLIP Spurious Correlation Pipeline.
----------------------------------------------------------

These tests validate the pipeline WITHOUT needing CLIP installed,
by mocking the model where needed. They check:

1. Dataset construction correctness
2. Colorization logic
3. Group ID calculation
4. Analysis functions
5. File saving

Run with:
    python -m pytest tests/test_pipeline.py -v
    # or simply:
    python tests/test_pipeline.py
"""

import sys
import os
import numpy as np
from PIL import Image

# Make sure we can import from the parent directory
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Colorization utility
# ─────────────────────────────────────────────────────────────────────────────

def test_colorize_image():
    """colorize_image should produce an RGB image with the correct color."""
    from datasets.colored_mnist import colorize_image

    gray = np.zeros((28, 28), dtype=np.uint8)
    gray[10:20, 10:20] = 255  # white square in center

    rgb = (200, 50, 30)
    result = colorize_image(gray, rgb)

    assert result.shape == (28, 28, 3), f"Expected (28,28,3), got {result.shape}"
    assert result.dtype == np.uint8, "Expected uint8"

    # Background (pixel 0,0) should be black
    assert result[0, 0].tolist() == [0, 0, 0], f"Background should be black, got {result[0,0]}"

    # Foreground pixel should be approximately the given color
    np.testing.assert_array_equal(result[15, 15], rgb,
        err_msg="Foreground pixel should match the target color")

    print("  ✓ test_colorize_image passed")


def test_colorize_partial_intensity():
    """Pixels with intermediate intensity should be partially colored."""
    from datasets.colored_mnist import colorize_image

    gray = np.full((8, 8), 128, dtype=np.uint8)  # mid-gray everywhere
    rgb  = (200, 100, 50)
    result = colorize_image(gray, rgb)

    expected = tuple((128 / 255.0 * c) for c in rgb)
    for c_idx, exp_val in enumerate(expected):
        assert abs(int(result[4, 4, c_idx]) - int(exp_val)) <= 1, \
            f"Channel {c_idx}: expected ~{exp_val:.0f}, got {result[4,4,c_idx]}"

    print("  ✓ test_colorize_partial_intensity passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Colored MNIST construction
# ─────────────────────────────────────────────────────────────────────────────

def test_colored_mnist_aligned_structure():
    """
    In aligned split, each sample's color should match its label's natural color.
    """
    from datasets.colored_mnist import ColoredMNIST, DIGIT_COLORS

    dataset = ColoredMNIST(root="./data", split="aligned", download=True, max_samples=100)

    assert len(dataset) == 100, f"Expected 100 samples, got {len(dataset)}"

    for pil_img, label, color_name, group_id in dataset:
        expected_color = DIGIT_COLORS[label][0]
        assert color_name == expected_color, \
            f"Label {label} should have color '{expected_color}', got '{color_name}'"

        expected_group_id = label * 10 + list(DIGIT_COLORS.keys()).index(label)
        assert isinstance(pil_img, Image.Image), "Should return PIL Image"
        assert pil_img.mode == "RGB", f"Image should be RGB, got {pil_img.mode}"

    print(f"  ✓ test_colored_mnist_aligned_structure passed ({len(dataset)} samples checked)")


def test_colored_mnist_misaligned_structure():
    """In misaligned split, each base image spawns 10 versions (one per color)."""
    from datasets.colored_mnist import ColoredMNIST, ALL_COLORS

    # Use 1 sample → should produce 10 colored versions (one per color)
    dataset = ColoredMNIST(root="./data", split="misaligned", download=True, max_samples=1)
    assert len(dataset) == 10, \
        f"With 1 base sample in misaligned split, expected 10 colors, got {len(dataset)}"

    seen_colors = set(color for _, _, color, _ in dataset)
    assert seen_colors == set(ALL_COLORS), \
        f"Expected all 10 colors, got: {seen_colors}"

    # All samples should have the same label
    labels = set(label for _, label, _, _ in dataset)
    assert len(labels) == 1, f"All misaligned versions should share same label, got: {labels}"

    print("  ✓ test_colored_mnist_misaligned_structure passed")


def test_group_ids_are_unique_per_label_color():
    """Group IDs should uniquely identify (label, color) pairs."""
    from datasets.colored_mnist import ColoredMNIST

    dataset = ColoredMNIST(root="./data", split="aligned", download=True, max_samples=200)

    seen = {}
    for _, label, color_name, group_id in dataset:
        key = (label, color_name)
        if key in seen:
            assert seen[key] == group_id, \
                f"Inconsistent group_id for {key}: {seen[key]} vs {group_id}"
        seen[key] = group_id

    # Different (label, color) pairs should have different group_ids
    group_ids = list(seen.values())
    assert len(group_ids) == len(set(group_ids)), \
        "Each (label, color) should have a unique group_id"

    print("  ✓ test_group_ids_are_unique_per_label_color passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 3: Colored CIFAR-10 construction
# ─────────────────────────────────────────────────────────────────────────────

def test_colored_cifar10_tint():
    """Tinting should produce an image that is a blend of original and color."""
    from datasets.colored_cifar10 import apply_tint

    original = np.full((32, 32, 3), 200, dtype=np.uint8)  # light gray
    rgb      = (0, 0, 255)  # pure blue
    alpha    = 0.5

    result = apply_tint(original, rgb, alpha=alpha)
    assert result.shape == (32, 32, 3)

    # Expected: 0.5 * 200 + 0.5 * 0   = 100  (red channel)
    #           0.5 * 200 + 0.5 * 0   = 100  (green channel)
    #           0.5 * 200 + 0.5 * 255 = 227  (blue channel)
    expected = [100, 100, 227]
    for c in range(3):
        assert abs(int(result[0, 0, c]) - expected[c]) <= 1, \
            f"Channel {c}: expected ~{expected[c]}, got {result[0,0,c]}"

    print("  ✓ test_colored_cifar10_tint passed")


def test_colored_cifar10_alpha_extremes():
    """alpha=1.0 should return original image; alpha=0.0 should return pure tint color."""
    from datasets.colored_cifar10 import apply_tint

    original = np.full((8, 8, 3), 150, dtype=np.uint8)
    rgb      = (10, 20, 30)

    result_no_tint = apply_tint(original, rgb, alpha=1.0)
    np.testing.assert_array_equal(result_no_tint, original,
        err_msg="alpha=1.0 should return the original image unchanged")

    result_full_tint = apply_tint(original, rgb, alpha=0.0)
    expected_full = np.full((8, 8, 3), rgb, dtype=np.uint8)
    np.testing.assert_array_equal(result_full_tint, expected_full,
        err_msg="alpha=0.0 should return a solid color fill")

    print("  ✓ test_colored_cifar10_alpha_extremes passed")


def test_colored_cifar10_aligned():
    """CIFAR-10 aligned split: each class should have its natural tint."""
    from datasets.colored_cifar10 import ColoredCIFAR10, CLASS_TINTS

    dataset = ColoredCIFAR10(root="./data", split="aligned", download=True, max_samples=50)
    assert len(dataset) == 50

    for pil_img, label, tint_name, group_id in dataset:
        expected_tint = CLASS_TINTS[label][0]
        assert tint_name == expected_tint, \
            f"Class {label} should have tint '{expected_tint}', got '{tint_name}'"
        assert isinstance(pil_img, Image.Image)
        assert pil_img.mode == "RGB"

    print(f"  ✓ test_colored_cifar10_aligned passed ({len(dataset)} samples checked)")


# ─────────────────────────────────────────────────────────────────────────────
# Test 4: Analysis functions (with mock inference results)
# ─────────────────────────────────────────────────────────────────────────────

def _make_mock_results(n=200, n_classes=10, dataset_name="mnist", perfect=False):
    """
    Create fake inference results to test analysis functions without CLIP.

    If perfect=True, predictions always match true labels.
    Otherwise, predictions are partially wrong with a color bias built in.
    """
    from datasets.colored_mnist import DIGIT_COLORS, ALL_COLORS

    rng = np.random.default_rng(0)
    true_labels  = rng.integers(0, n_classes, size=n)
    color_names  = [DIGIT_COLORS[label][0] for label in true_labels]  # aligned
    group_ids    = np.array([label * 10 + ALL_COLORS.index(color)
                             for label, color in zip(true_labels, color_names)])

    if perfect:
        preds_shape = true_labels.copy()
        preds_color = true_labels.copy()
    else:
        # Mostly correct, but 30% wrong
        preds_shape = true_labels.copy()
        wrong_mask  = rng.random(n) < 0.3
        preds_shape[wrong_mask] = rng.integers(0, n_classes, size=wrong_mask.sum())

        preds_color = true_labels.copy()
        wrong_mask2 = rng.random(n) < 0.2
        preds_color[wrong_mask2] = rng.integers(0, n_classes, size=wrong_mask2.sum())

    shape_prompts = {i: f"shape prompt {i}" for i in range(n_classes)}
    color_prompts = {i: f"color prompt {i}" for i in range(n_classes)}

    return {
        "predictions_shape": preds_shape,
        "predictions_color": preds_color,
        "true_labels":       true_labels,
        "color_names":       color_names,
        "group_ids":         group_ids,
        "prompt_mode":       "both",
        "dataset_name":      dataset_name,
        "shape_prompts":     shape_prompts,
        "color_prompts":     color_prompts,
    }


def test_per_group_accuracy_shape():
    """per-group accuracy DataFrame should have expected columns and valid accuracy values."""
    from analysis import compute_per_group_accuracy

    results = _make_mock_results(n=500)
    df = compute_per_group_accuracy(results, "shape")

    expected_cols = {"label", "label_name", "color", "is_aligned", "n_samples", "n_correct", "accuracy"}
    assert expected_cols.issubset(set(df.columns)), \
        f"Missing columns: {expected_cols - set(df.columns)}"

    assert (df["accuracy"] >= 0).all() and (df["accuracy"] <= 100).all(), \
        "Accuracy values must be in [0, 100]"
    assert (df["n_samples"] > 0).all(), "All groups must have at least one sample"

    print(f"  ✓ test_per_group_accuracy_shape passed ({len(df)} groups found)")


def test_perfect_predictor_accuracy():
    """A perfect predictor should give 100% accuracy in all groups."""
    from analysis import compute_per_group_accuracy, compute_summary_stats

    results = _make_mock_results(n=200, perfect=True)
    df = compute_per_group_accuracy(results, "shape")

    assert (df["accuracy"] == 100.0).all(), \
        f"Perfect predictor should give 100% everywhere. Min: {df['accuracy'].min()}"

    stats = compute_summary_stats(results, "shape")
    assert stats["overall_accuracy"] == 100.0
    assert stats["worst_group_accuracy"] == 100.0

    print("  ✓ test_perfect_predictor_accuracy passed")


def test_confusion_matrix_dimensions():
    """Confusion matrix should be square with side = n_classes."""
    from analysis import compute_confusion_matrix_data

    results   = _make_mock_results(n=300, n_classes=10)
    cm        = compute_confusion_matrix_data(results, "shape")

    assert cm.shape == (10, 10), f"Expected (10,10), got {cm.shape}"
    assert cm.sum() == 300,       f"Total counts should equal n_samples: {cm.sum()}"
    assert (cm >= 0).all(),       "All confusion matrix values should be non-negative"

    print("  ✓ test_confusion_matrix_dimensions passed")


def test_save_report_creates_file(tmp_path):
    """save_human_readable_report should create a non-empty .txt file."""
    from analysis import save_human_readable_report

    results  = _make_mock_results(n=100)
    out_dir  = str(tmp_path)
    filepath = save_human_readable_report(results, out_dir)

    assert os.path.isfile(filepath), f"Report file not found: {filepath}"
    assert os.path.getsize(filepath) > 500, "Report file seems too small — check content"

    with open(filepath) as f:
        content = f.read()
    assert "CLIP SPURIOUS CORRELATION ANALYSIS" in content
    assert "PER-GROUP ACCURACY TABLE" in content
    assert "CONFUSION MATRIX" in content

    print(f"  ✓ test_save_report_creates_file passed (size: {os.path.getsize(filepath)} bytes)")


def test_save_csv_creates_files(tmp_path):
    """save_csv_tables should create one CSV per prompt mode."""
    from analysis import save_csv_tables
    import pandas as pd

    results  = _make_mock_results(n=100)
    out_dir  = str(tmp_path)
    paths    = save_csv_tables(results, out_dir)

    assert len(paths) == 2, f"Expected 2 CSV files (shape + color), got {len(paths)}"
    for path in paths:
        assert os.path.isfile(path), f"CSV not found: {path}"
        df = pd.read_csv(path)
        assert len(df) > 0, "CSV should not be empty"
        assert "accuracy" in df.columns

    print("  ✓ test_save_csv_creates_files passed")


# ─────────────────────────────────────────────────────────────────────────────
# Test 5: Dataset get_group_info
# ─────────────────────────────────────────────────────────────────────────────

def test_get_group_info_aligned():
    """get_group_info for aligned split should have exactly 10 groups (one per digit)."""
    from datasets.colored_mnist import ColoredMNIST

    dataset    = ColoredMNIST(root="./data", split="aligned", download=True, max_samples=200)
    group_info = dataset.get_group_info()

    # Aligned: each digit has exactly 1 color → at most 10 groups
    assert len(group_info) <= 10, \
        f"Aligned split should have ≤10 groups, got {len(group_info)}"

    for (label, color), count in group_info.items():
        assert count > 0, f"Group ({label}, {color}) has 0 samples"

    print(f"  ✓ test_get_group_info_aligned passed ({len(group_info)} groups)")


# ─────────────────────────────────────────────────────────────────────────────
# Runner
# ─────────────────────────────────────────────────────────────────────────────

def run_all_tests():
    print("\n" + "=" * 60)
    print("  CLIP PIPELINE — SANITY TESTS")
    print("=" * 60 + "\n")

    import tempfile

    tests_no_args = [
        test_colorize_image,
        test_colorize_partial_intensity,
        test_colored_cifar10_tint,
        test_colored_cifar10_alpha_extremes,
        test_per_group_accuracy_shape,
        test_perfect_predictor_accuracy,
        test_confusion_matrix_dimensions,
        test_get_group_info_aligned,
    ]

    # Tests that need datasets downloaded
    tests_need_data = [
        test_colored_mnist_aligned_structure,
        test_colored_mnist_misaligned_structure,
        test_group_ids_are_unique_per_label_color,
        test_colored_cifar10_aligned,
    ]

    # Tests that need a tmp_path
    tests_need_tmpdir = [
        test_save_report_creates_file,
        test_save_csv_creates_files,
    ]

    passed = 0
    failed = 0

    all_tests = (
        [(t, []) for t in tests_no_args]
        + [(t, []) for t in tests_need_data]
        + [(t, [tempfile.mkdtemp()]) for t in tests_need_tmpdir]
    )

    for test_fn, extra_args in all_tests:
        try:
            test_fn(*extra_args)
            passed += 1
        except Exception as e:
            print(f"  ✗ {test_fn.__name__} FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"  Results: {passed} passed, {failed} failed")
    if failed == 0:
        print("  All tests passed! Pipeline is ready to run.")
    else:
        print("  Some tests failed. Check output above.")
    print("=" * 60 + "\n")
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
