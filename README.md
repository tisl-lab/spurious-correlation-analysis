# CLIP Spurious Correlation Analysis — Colored MNIST

## What This Project Does

This pipeline uses **OpenAI's CLIP** (ViT-B/32) to investigate **spurious correlations**
in computer vision. The central question:

> *Does CLIP classify images based on their true semantic content (shape, digit identity),
> or does it exploit spurious shortcuts like background color?*

We test this on **Colored MNIST** — handwritten digits where each digit class is
spuriously correlated with a background color (red / green / blue). CLIP is evaluated
both **zero-shot** (no fine-tuning) and after **partial fine-tuning** on a labeled
subset, sweeping fine-tune percentages from 0% to 90% to see how supervision shifts
the model's reliance on shape vs. color.

---

## Project Structure

```
clip-sp-rel/
├── README.md                       ← You are here
├── requirements.txt                ← Python dependencies
│
├── run_experiment.py               ← Main entry point (single run)
├── run_sweep.sh                    ← Local sweep: 0%→90% fine-tune, then visualize
├── visualize_sweep.py              ← Reads sweep CSVs, produces sweep_summary.png
│
├── clip_zero_shot.py               ← CLIP wrapper: zero-shot inference + fine-tuning
├── analysis.py                     ← Per-group accuracy helpers
│
├── datasets/
│   ├── colored_mnist.py            ← Colored MNIST dataset (red/green/blue backgrounds)
│   └── colored_cifar10.py          ← Colored CIFAR-10 dataset
│
├── tests/
│   ├── test_finetune_diagnostics.py ← Checks dtype stability, NaN detection, loss
│   ├── test_pipeline.py            ← End-to-end pipeline sanity checks
│   └── test_sample.py              ← Dataset loading checks
│
├── cc_setup.sh                     ← One-time setup on Compute Canada (login node)
└── submit_sweep.sh                 ← SLURM array job: runs all 10 fractions in parallel
```

---

## Key Concepts

### Spurious Correlation
A spurious correlation is a shortcut a model learns that works in training but fails to
generalize. Here, if all "0" digits appear on red backgrounds, a model may learn
"red → 0" rather than "round shape → 0". When the background color changes, accuracy
drops — revealing the shortcut.

### Dataset Design
Colored MNIST assigns a background color to each digit class in a fixed cycle:

| Digit | Aligned color |
|-------|--------------|
| 0, 3, 6, 9 | Red   |
| 1, 4, 7    | Green |
| 2, 5, 8    | Blue  |

Each image's background color is encoded in its filename (`red_1234.png`, etc.) —
no pixel-level color analysis needed.

### Two Prompt Strategies

| Mode | Example prompt | What it tests |
|------|----------------|---------------|
| `shape` | "a photo of the handwritten digit zero" | Does CLIP use digit shape? |
| `color` | "a photo of a handwritten digit on a red background" | Does CLIP use background color? |

Comparing accuracy under each mode — especially on misaligned images — reveals
whether CLIP relies on shape or color as its primary cue.

### Fine-Tuning Protocol
Only the CLIP **image encoder** is updated during fine-tuning; the text encoder stays
frozen. Classification logits are still computed as cosine similarity between image and
text features (× logit scale), so fine-tuned and zero-shot results are directly
comparable.

> **MPS note**: Apple Silicon MPS has known autograd instabilities with CLIP's ViT
> transformer. Fine-tuning automatically falls back to CPU on MPS devices and returns
> the model to MPS for inference.

---

## Quick Start

```bash
# 1. Install CLIP
pip install git+https://github.com/openai/CLIP.git

# 2. Install other dependencies
pip install -r requirements.txt

# 3. Pure zero-shot (no fine-tuning)
python run_experiment.py

# 4. With fine-tuning (e.g. 30% of data used to fine-tune)
python run_experiment.py --fine_tune_pct 0.3

# 5. Quick smoke test
python run_experiment.py --max_samples 200 --fine_tune_pct 0.4 --ft_epochs 1
```

### All CLI flags

```bash
python run_experiment.py \
    --clip_model    ViT-B/32        # model variant (ViT-B/32 | ViT-B/16 | ViT-L/14 | RN50 | RN101)
    --data_dir      ./data          # path to Colored MNIST data
    --output_dir    ./results/mnist # where CSVs are saved
    --max_samples   None            # cap total images (default: all)
    --batch_size    64              # inference / training batch size
    --seed          42              # random seed
    --fine_tune_pct 0.0             # fraction used for fine-tuning (0.0 = pure zero-shot)
    --ft_epochs     3               # fine-tuning epochs
    --ft_lr         1e-5            # fine-tuning learning rate
```

---

## Fine-Tuning Sweep (0% → 90%)

Run all ten fine-tune fractions locally and generate a visual summary:

```bash
bash run_sweep.sh
# Saves results to results/ft_sweep/pct_{0,10,...,90}/
# Generates results/ft_sweep/sweep_summary.png and sweep_summary.csv
```

The 4-panel figure shows:
1. **Overall accuracy** vs fine-tune percentage
2. **Per-digit accuracy** curves
3. **Per-background-color accuracy** (red / green / blue)
4. **Summary table** with Δ vs zero-shot baseline

---

## Running on Compute Canada (HPC)

### One-time setup (run on login node)
```bash
bash cc_setup.sh
```
This loads modules (`python/3.10`, `cuda/11.8`, `cudnn`), creates a virtualenv in
`$PROJECT/clip_env`, installs all dependencies, and pre-downloads CLIP weights so
compute nodes don't need internet access.

### Parallel sweep with SLURM
```bash
# Edit submit_sweep.sh: set --account=YOUR_ACCOUNT
mkdir -p logs
sbatch submit_sweep.sh
```
This submits a SLURM array job with one task per fine-tune percentage (0%, 10%, …, 90%),
all running in parallel on separate GPUs. Results land in
`results/ft_sweep/pct_{0,10,...,90}/`.

---

## Understanding the Output

Each run produces three CSVs in `--output_dir`:

| File | Contents |
|------|----------|
| `per_digit_*.csv` | Accuracy per digit (0–9) |
| `per_color_*.csv` | Accuracy per background color (red / green / blue) |
| `per_digit_per_color_*.csv` | Accuracy for every (digit, color) pair |

Console output example (zero-shot):
```
══════════════════════════════════════════════════════════
  OVERALL ACCURACY: 8541/10000 = 85.41%
══════════════════════════════════════════════════════════

  PER-DIGIT / PER-BACKGROUND-COLOR ACCURACY
  Digit          blue     green       red
  ────────────────────────────────────────
  0             82.30%   79.10%    91.20%
  1             88.50%   93.40%    85.60%
  ...
```

A large accuracy drop on a particular background color for a given digit indicates
CLIP is influenced by that spurious color cue rather than digit shape alone.

---

## Diagnostics & Tests

```bash
# Check fine-tuning stability (dtype, NaN detection, loss convergence)
python tests/test_finetune_diagnostics.py

# End-to-end pipeline checks
pytest tests/ -v -s
```
