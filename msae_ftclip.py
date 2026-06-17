"""
msae_ftclip.py
==============

Load the fine-tuned CLIP model and exact image lists saved by
run_waterbirds_msae.py, then extract features for SAE training.

The run folder is located by reconstructing the naming used in
run_waterbirds_msae.py:
    <output_dir>/clip_ft_{BIASED|FULL}_{max_samples}_<timestamp>/

Pass the same --biased_ft / --max_samples / --output_dir values that were
used when training, and this script will find the matching run automatically.
If multiple matching runs exist, the most recent one is used.

Output .npy files are written inside the run folder, named:
    waterbirds_ft_fttrain_{N}_{D}.npy
    waterbirds_ft_test_{N}_{D}.npy

(SAEDataset in msae/utils.py parses N and D from the filename stem.)

USAGE:
    python msae_ftclip.py
    python msae_ftclip.py --max_samples 400 --biased_ft --output_dir ./results/waterbirds
    python msae_ftclip.py --run_dir results/waterbirds/clip_ft_BIASED_400_20260602_161950
"""

import argparse
import glob
import os
import sys
import inspect  ## for context_decorator for ablation hooks

_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "msae"))

import metrics as msae_metrics  # type: ignore[import]
from clip_zero_shot import CLIPZeroShot

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader


# ── Manifest-backed dataset ────────────────────────────────────────────────────

class ManifestDataset:
    """Dataset built from a saved manifest CSV — exact same images as the run."""
    clip_preprocess = None

    def __init__(self, manifest_path: str):
        df = pd.read_csv(manifest_path)
        self.samples = [
            (row["img_path"], int(row["label"]), row["bg"], int(row["group_id"]))
            for _, row in df.iterrows()
        ]
        self.manifest_path = manifest_path

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        from PIL import Image
        path, label, bg_name, gid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.clip_preprocess is not None:
            img = self.clip_preprocess(img)
        return img, label, bg_name, gid


# ── CLI — mirrors run_waterbirds_msae.py ───────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract fine-tuned CLIP features from a saved run for SAE training"
    )
    # ── same args and defaults as run_waterbirds_msae.py ─────────────────────
    parser.add_argument("--dataset",      type=str, default="waterbirds",
                        help="Dataset name used in file/folder naming.")
    parser.add_argument("--clip_model",  type=str, default="ViT-B/32",
                        choices=["ViT-B/32", "ViT-B/16", "ViT-L/14", "RN50", "RN101"])
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Results directory (default: ./results/<dataset>)")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ft_epochs",   type=int, default=3)
    parser.add_argument("--ft_lr",       type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--biased_ft",   action="store_true", default=True)
    # ── optional explicit override ────────────────────────────────────────────
    parser.add_argument("--run_dir",     type=str, default=None,
                        help="Explicit path to a clip_ft_*/ run folder. "
                             "If omitted, matched from the other args via model metadata.")
    # parser.add_argument("--sae_path",    type=str, default=None,
    #                     help="Path to a trained SAE .pth file (msae/train.py output). "
    #                          "If provided, representations are extracted after feature extraction.")
    parser.add_argument("--sae_path",    type=str, default=None,
                        help="Path to a trained SAE .pth file (msae/train.py output). "
                             "If provided, representations are extracted after feature extraction.")
    # ── SAE architecture — mirrors msae/train.py ─────────────────────────────
    parser.add_argument("-m", "--model",      type=str, default="MSAE_RW",
                        choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW"],
                        help="SAE model architecture (matches msae/train.py --model)")
    parser.add_argument("-a", "--activation", type=str, default="",
                        help="SAE activation string (matches msae/train.py --activation, "
                             "e.g. 'ReLU_03', 'TopKReLU_64')")
    return parser.parse_args()


# ── Run-folder resolution ──────────────────────────────────────────────────────

def resolve_run_dir(args):
    """
    Find the saved run folder whose model.pt metadata matches the current args.

    Reads the metadata dict stored inside every clip_ft_*/model.pt and compares
    ft_tag, max_samples, ft_epochs, ft_lr, seed, and clip_model — the same
    fields written by run_waterbirds_msae.py.  Returns (run_dir, metadata).
    """
    if args.run_dir is not None:
        if not os.path.isdir(args.run_dir):
            raise FileNotFoundError(f"--run_dir not found: {args.run_dir}")
        pt = os.path.join(args.run_dir, "model.pt")
        payload  = torch.load(pt, map_location="cpu", weights_only=False)
        return args.run_dir, payload.get("metadata", {})

    ft_tag = "BIASED" if args.biased_ft else "FULL"

    candidates = glob.glob(
        os.path.join(args.output_dir, "clip_ft_*", "model.pt")
    )

    matches = []
    for model_pt in candidates:
        payload = torch.load(model_pt, map_location="cpu", weights_only=False)
        meta    = payload.get("metadata", {})
        if (meta.get("ft_tag")      == ft_tag           and
            meta.get("max_samples") == args.max_samples and
            meta.get("seed")        == args.seed):
            run_dir = os.path.dirname(model_pt)
            matches.append((run_dir, meta))

    if not matches:
        raise FileNotFoundError(
            f"No saved model matches: ft_tag={ft_tag}, max_samples={args.max_samples}, "
            f"seed={args.seed}\n"
            f"Searched in: {args.output_dir}\n"
            f"Run run_waterbirds_msae.py with matching parameters first."
        )

    if len(matches) > 1:
        print(f"  Found {len(matches)} matching runs — using most recent.")
    best_run_dir, best_meta = max(
        matches, key=lambda x: os.path.getmtime(os.path.join(x[0], "model.pt"))
    )
    return best_run_dir, best_meta


# ── Feature extraction ─────────────────────────────────────────────────────────

@torch.no_grad()
def extract_features(clip_model, dataset, batch_size, device):
    dataset.clip_preprocess = clip_model.preprocess

    def collate(batch):
        imgs      = torch.stack([b[0] for b in batch])
        labels    = torch.tensor([b[1] for b in batch], dtype=torch.long)
        bg_names  = [b[2] for b in batch]
        group_ids = torch.tensor([b[3] for b in batch], dtype=torch.long)
        return imgs, labels, bg_names, group_ids

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, collate_fn=collate)

    all_features = []
    for imgs, _, _, _ in tqdm(loader, desc="  Extracting"):
        imgs  = imgs.to(device)
        feats = clip_model.model.encode_image(imgs).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        all_features.append(feats.cpu().numpy())

    return np.concatenate(all_features, axis=0).astype(np.float32)


def save_memmap(features, path):
    n, d = features.shape
    mm = np.memmap(path, dtype="float32", mode="w+", shape=(n, d))
    mm[:] = features
    mm.flush()
    print(f"  Saved: {path}  [{n} × {d}]")
    return path


# ── SAE representation extraction ─────────────────────────────────────────────

@torch.no_grad()
def extract_sae_representations(clip_model, sae_model, dataset, device):
    """
    Pass every image through the fine-tuned CLIP encoder then the SAE.

    Returns a dict with:
        clip_representations  : Tensor (N, D)  — L2-normalised CLIP features
        sae_representations   : Tensor (N, L)  — sparse SAE latents (full)
        sae_reconstructed     : Tensor (N, D)  — SAE reconstruction of CLIP features
        image_paths           : list[str]
        metrics               : list[dict]     — per-image fvu / mae / cs / l0 / highest_magnitude
    """
    import torch.nn.functional as F

    clip_model.model.eval()
    sae_model.eval()

    clip_reps, sae_reps, sae_recons = [], [], []
    image_paths, metrics_all = [], []

    for path, _, _, _ in tqdm(dataset.samples, desc="  SAE extract"):
        from PIL import Image
        img   = Image.open(path).convert("RGB")
        inp   = clip_model.preprocess(img).unsqueeze(0).to(device)

        features = clip_model.model.encode_image(inp).float()           # (1, D)
        # post_reconstructed, reconstructed, full_latents
        post_reconstructed, _, full_latents = sae_model(features)    # Autoencoder forward
        
        clip_reps.append(features.cpu().flatten())
        sae_reps.append(full_latents.cpu().flatten())
        sae_recons.append(post_reconstructed.cpu().flatten())
        image_paths.append(path)

        fvu = msae_metrics.explained_variance(features, post_reconstructed)
        mae = msae_metrics.normalized_mean_absolute_error(features, post_reconstructed)
        cs  = F.cosine_similarity(features, post_reconstructed)
        l0  = msae_metrics.l0_messure(full_latents)
        highest_magnitude = full_latents.max(dim=-1).values

        metrics_all.append({
            "fvu":               fvu.item() if hasattr(fvu, "item") else float(fvu),
            "mae":               mae.item(),
            "cs":                cs.mean().item(),
            "l0":                l0.item(),
            "highest_magnitude": highest_magnitude.mean().item(),
        })

    return {
        "clip_representations": torch.stack(clip_reps),
        "sae_representations":  torch.stack(sae_reps),
        "sae_reconstructed":    torch.stack(sae_recons),
        "image_paths":          image_paths,
        "metrics":              metrics_all,
    }


def find_top_concept_images(results, top_k=5, concept_index=None,
                            mean_activations=None, concept_match_scores=None,
                            vocab_names=None, save_path=None):
    """
    Find images with the largest concept activations.

    Args:
        results              : dict from extract_sae_representations
        top_k                : number of top images to return (default 5)
        concept_index        : rank by this specific SAE latent; None → peak across all latents
        mean_activations     : (L,) tensor; subtracted before ranking when provided
        concept_match_scores : (n_vocab, n_latents) array for concept naming
        vocab_names          : list of vocab word strings
        save_path            : if set (and concept_match_scores/vocab_names given), saves a
                               2-panel figure (top-1 image + top-10 concept bar chart)

    Returns:
        list of dicts, length top_k, each with:
            image_index, image_path, concept_index, activation
    """
    sae_reps = results["sae_representations"]   # (N, L)
    paths    = results["image_paths"]

    reps = sae_reps if mean_activations is None else (sae_reps - mean_activations)

    if concept_index is not None:
        scores = reps[:, concept_index]
        top    = torch.topk(scores, min(top_k, len(scores)))
        hits = [
            {
                "image_index":   idx.item(),
                "image_path":    paths[idx.item()],
                "concept_index": concept_index,
                "activation":    val.item(),
            }
            for val, idx in zip(top.values, top.indices)
        ]
    else:
        peak_vals, peak_concepts = reps.max(dim=1)
        top = torch.topk(peak_vals, min(top_k, len(peak_vals)))
        hits = [
            {
                "image_index":   idx.item(),
                "image_path":    paths[idx.item()],
                "concept_index": peak_concepts[idx].item(),
                "activation":    val.item(),
            }
            for val, idx in zip(top.values, top.indices)
        ]

    if save_path is not None and concept_match_scores is not None and vocab_names is not None:
        import matplotlib.pyplot as plt
        from PIL import Image as _PILImage
        top1     = hits[0]
        img_idx  = top1["image_index"]
        top_10   = reps[img_idx].topk(10)
        names    = [vocab_names[concept_match_scores[:, i].argmax()] + f"/{i}"
                    for i in top_10.indices.cpu().numpy()]
        fig, ax  = plt.subplots(1, 2, figsize=(10, 4))
        ax[0].imshow(_PILImage.open(paths[img_idx]).convert("RGB"))
        ax[0].axis("off")
        ax[0].set_title(f"Image {img_idx}")
        ax[1].barh(range(10), top_10.values.cpu().numpy())
        ax[1].set_yticks(range(10))
        ax[1].set_yticklabels(names)
        ax[1].set_title(f"Top 10 SAE Concepts for Image {img_idx}")
        plt.tight_layout()
        plt.savefig(save_path)
        # plt.show()

    return hits


GROUP_NAMES = {
    0: "landbird / land bg",
    1: "landbird / water bg",
    2: "waterbird / land bg",
    3: "waterbird / water bg",
}


def _clip_predict(clip_ft, path, device):
    """Zero-shot CLIP classification → 'landbird' or 'waterbird'."""
    import clip as openai_clip
    from PIL import Image as _PIL
    CLASS_PROMPTS = ["a photo of a landbird", "a photo of a waterbird"]
    CLASS_LABELS  = ["landbird", "waterbird"]
    tokens = openai_clip.tokenize(CLASS_PROMPTS).to(device)
    with torch.no_grad():
        txt  = clip_ft.model.encode_text(tokens).float()
        txt  = txt / txt.norm(dim=-1, keepdim=True)
        img  = clip_ft.preprocess(_PIL.open(path).convert("RGB")).unsqueeze(0).to(device)
        feat = clip_ft.model.encode_image(img).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        pred = (feat @ txt.T).squeeze(0).argmax().item()
    return CLASS_LABELS[pred]


def plot_misaligned_concept_images(
    results, group_ids, group_top_sets, group_prevalence,
    concept_match_scores, vocab_names, sae_dir,
    clip_ft=None, device=None,
    top_n_concepts=3, top_k_images=5,
):
    """
    For each pair (misaligned group, aligned group):
      - take the top_n_concepts shared high-magnitude concepts (ranked by prevalence)
      - plot top_k_images images per concept per group (misaligned row first,
        aligned row below it)
      - for misaligned rows, show the fine-tuned CLIP prediction under each image

    Pairs:  group 1 (landbird/water, misaligned) ↔ group 3 (waterbird/water, aligned)
            group 2 (waterbird/land, misaligned) ↔ group 0 (landbird/land, aligned)
    """
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL

    sae_reps      = results["sae_representations"]   # (N, L)
    paths         = results["image_paths"]
    group_ids_arr = np.array(group_ids)

    for gid_mis, gid_aln in [(1, 3), (2, 0)]:
        shared_ids = sorted(
            group_top_sets[gid_mis] & group_top_sets[gid_aln],
            key=lambda c: -(group_prevalence[gid_mis][c] + group_prevalence[gid_aln][c]),
        )[:top_n_concepts]

        if not shared_ids:
            print(f"No shared concepts found for groups {gid_mis} & {gid_aln} — skipping plot.")
            continue

        n_rows = len(shared_ids) * 2          # 2 rows per concept
        fig, axes = plt.subplots(
            n_rows, top_k_images,
            figsize=(top_k_images * 2.8, n_rows * 3.2),
            squeeze=False,
        )
        fig.suptitle(
            f"Shared Concepts — misaligned: Group {gid_mis} ({GROUP_NAMES[gid_mis]})"
            f"  |  aligned: Group {gid_aln} ({GROUP_NAMES[gid_aln]})",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for ci, cid in enumerate(shared_ids):
            cname = vocab_names[concept_match_scores[:, cid].argmax()]

            for row_offset, (gid, is_mis) in enumerate([(gid_mis, True), (gid_aln, False)]):
                row = ci * 2 + row_offset

                # indices in results belonging to this group
                g_indices = np.where(group_ids_arr == gid)[0]
                g_acts    = sae_reps[g_indices, cid]
                top_local = torch.topk(g_acts, min(top_k_images, len(g_indices)))
                top_glob  = g_indices[top_local.indices.cpu().numpy()]

                # left-most cell carries the row label (matches report format)
                axes[row, 0].text(
                    -0.05, 0.5,
                    f"Grp{gid} ({GROUP_NAMES[gid]}) | {cname}/#{cid}",
                    transform=axes[row, 0].transAxes,
                    fontsize=7, rotation=0, ha="right", va="center",
                    clip_on=False,
                )

                for col, idx in enumerate(top_glob):
                    ax  = axes[row, col]
                    act = sae_reps[idx, cid].item()
                    ax.imshow(_PIL.open(paths[idx]).convert("RGB"))
                    ax.axis("off")
                    if is_mis and clip_ft is not None:
                        pred  = _clip_predict(clip_ft, paths[idx], device)
                        title = f"pred: {pred}\nact={act:.2f}"
                    else:
                        title = f"act={act:.2f}"
                    ax.set_title(title, fontsize=7)

                for col in range(len(top_glob), top_k_images):
                    axes[row, col].axis("off")

        plt.tight_layout()
        fname = f"misaligned_shared_concepts_groups_{gid_mis}_{gid_aln}.png"
        fpath = os.path.join(sae_dir, fname)
        plt.savefig(fpath, bbox_inches="tight")
        # plt.show()
        print(f"Saved: {fpath}")


def plot_misaligned_vs_fttrain_concept_images(
    te_results, ft_results, group_ids_te, group_top_sets, group_prevalence,
    concept_match_scores, vocab_names, sae_dir,
    clip_ft=None, device=None,
    top_n_concepts=3, top_k_images=5,
):
    """
    For each misaligned/aligned pair, show the top shared concepts as a
    2-row-per-concept grid:
      Row 1 — misaligned group images from the TEST set  (+ CLIP prediction)
      Row 2 — top activating images from the FINE-TUNE set for the same concept

    Saves one figure per pair:
        misaligned_vs_fttrain_groups_{gid_mis}_{gid_aln}.png
    """
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL

    te_reps       = te_results["sae_representations"]    # (N_te, L)
    te_paths      = te_results["image_paths"]
    ft_reps       = ft_results["sae_representations"]    # (N_ft, L)
    ft_paths      = ft_results["image_paths"]
    group_ids_arr = np.array(group_ids_te)

    for gid_mis, gid_aln in [(1, 3), (2, 0)]:
        shared_ids = sorted(
            group_top_sets[gid_mis] & group_top_sets[gid_aln],
            key=lambda c: -(int(group_prevalence[gid_mis][c]) + int(group_prevalence[gid_aln][c])),
        )[:top_n_concepts]

        if not shared_ids:
            print(f"No shared concepts for groups {gid_mis} & {gid_aln} — skipping ft plot.")
            continue

        n_rows = len(shared_ids) * 2
        fig, axes = plt.subplots(
            n_rows, top_k_images,
            figsize=(top_k_images * 2.8, n_rows * 3.2),
            squeeze=False,
        )
        fig.suptitle(
            f"Misaligned (test) vs Fine-tune set — "
            f"Group {gid_mis} ({GROUP_NAMES[gid_mis]}) misaligned",
            fontsize=11, fontweight="bold", y=1.01,
        )

        for ci, cid in enumerate(shared_ids):
            cname = vocab_names[concept_match_scores[:, cid].argmax()]

            # ── Row 0: misaligned group from TEST set ────────────────────────
            row_mis = ci * 2
            g_idx   = np.where(group_ids_arr == gid_mis)[0]
            g_acts  = te_reps[g_idx, cid]
            top_loc = torch.topk(g_acts, min(top_k_images, len(g_idx)))
            top_g   = g_idx[top_loc.indices.cpu().numpy()]

            axes[row_mis, 0].text(
                -0.05, 0.5,
                f"Grp{gid_mis} ({GROUP_NAMES[gid_mis]}) | {cname}/#{cid}",
                transform=axes[row_mis, 0].transAxes,
                fontsize=7, rotation=0, ha="right", va="center", clip_on=False,
            )
            for col, idx in enumerate(top_g):
                ax  = axes[row_mis, col]
                act = te_reps[idx, cid].item()
                ax.imshow(_PIL.open(te_paths[idx]).convert("RGB"))
                ax.axis("off")
                if clip_ft is not None:
                    pred  = _clip_predict(clip_ft, te_paths[idx], device)
                    title = f"pred: {pred}\nact={act:.2f}"
                else:
                    title = f"act={act:.2f}"
                ax.set_title(title, fontsize=7)
            for col in range(len(top_g), top_k_images):
                axes[row_mis, col].axis("off")

            # ── Row 1: top activating images from FINE-TUNE set ─────────────
            row_ft  = ci * 2 + 1
            ft_acts = ft_reps[:, cid]
            top_ft  = torch.topk(ft_acts, min(top_k_images, len(ft_acts)))

            axes[row_ft, 0].text(
                -0.05, 0.5,
                f"ft-train | {cname}/#{cid}",
                transform=axes[row_ft, 0].transAxes,
                fontsize=7, rotation=0, ha="right", va="center", clip_on=False,
            )
            for col, idx in enumerate(top_ft.indices.cpu().numpy()):
                ax  = axes[row_ft, col]
                act = ft_reps[idx, cid].item()
                ax.imshow(_PIL.open(ft_paths[idx]).convert("RGB"))
                ax.axis("off")
                ax.set_title(f"act={act:.2f}", fontsize=7)
            for col in range(len(top_ft.indices), top_k_images):
                axes[row_ft, col].axis("off")

        plt.tight_layout()
        fname = f"misaligned_vs_fttrain_groups_{gid_mis}_{gid_aln}.png"
        fpath = os.path.join(sae_dir, fname)
        plt.savefig(fpath, bbox_inches="tight")
        # plt.show()
        print(f"Saved: {fpath}")


def analyze_group_concepts(results, group_ids, sae_model, concept_match_scores,
                            vocab_names, sae_dir, top_k=50,
                            activation_threshold=None,
                            clip_ft=None, device=None,
                            ft_results=None):
    """
    Plot concept distribution across groups, save a shared-concept report,
    and visualise misaligned-vs-aligned image rows for top shared concepts.

    Concepts are ranked by **prevalence**: the number of images in the group
    where the concept activation exceeds activation_threshold.  This avoids
    concepts that are very high for a handful of images but rare overall.

    Args:
        results              : dict with 'sae_representations' (N, L) tensor
        group_ids            : list[int] of group label per image (len N)
        sae_model            : SAE model (used for latent_dim)
        concept_match_scores : (n_vocab, n_latents) numpy array
        vocab_names          : list of vocab word strings (len n_vocab)
        sae_dir              : output directory for plots and report
        top_k                : how many top-prevalence concepts per group (default 50)
        activation_threshold : activations above this count as "active"; when None
                               (default) the mean activation across all images and
                               concepts is used automatically
        clip_ft              : CLIPZeroShot model for misaligned predictions (optional)
        device               : torch device string (optional)
    """
    import matplotlib.pyplot as plt

    sae_reps      = results["sae_representations"]           # (N, L)
    group_ids_arr = np.array(group_ids)
    sae_reps_np   = sae_reps.cpu().numpy()

    if activation_threshold is None:
        activation_threshold = float(sae_reps_np.mean())
        print(f"  activation_threshold set to mean activation: {activation_threshold:.6f}")

    # ── Distribution plot ────────────────────────────────────────────────────
    highest_concepts = sae_reps.argmax(dim=1).cpu().numpy()
    plt.figure(figsize=(10, 5))
    plt.title("Distribution of Highest-Magnitude Concepts Across Groups")
    for gid in sorted(set(group_ids)):
        mask  = group_ids_arr == gid
        label = f"Group {gid} — {GROUP_NAMES.get(gid, str(gid))}"
        plt.hist(highest_concepts[mask], bins=np.arange(sae_model.latent_dim + 1) - 0.5,
                 alpha=0.5, label=label)
    plt.xlabel("Concept Index (SAE Latent Dimension)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(sae_dir, "concept_distribution_by_group.png"))
    # plt.show()

    # ── Per-group concept prevalence (# images above threshold) ─────────────
    group_prevalence = {}
    group_top_sets   = {}
    for gid in sorted(set(group_ids)):
        g_reps               = sae_reps_np[group_ids_arr == gid]          # (n_g, L)
        prev                 = (g_reps > activation_threshold).sum(axis=0) # (L,) int counts
        group_prevalence[gid]= prev
        top_idx              = np.argsort(prev)[::-1][:top_k]
        group_top_sets[gid]  = set(top_idx.tolist())

    # ── Shared-concept report ────────────────────────────────────────────────
    report_lines = [
        f"Shared High-Magnitude Concepts  "
        f"(top {top_k} per group by prevalence, threshold={activation_threshold})",
        "=" * 80,
    ]
    for gid_a, gid_b in [(0, 2), (3, 1)]:
        shared_ids = sorted(
            group_top_sets[gid_a] & group_top_sets[gid_b],
            key=lambda c: -(int(group_prevalence[gid_a][c]) + int(group_prevalence[gid_b][c])),
        )
        report_lines.append(
            f"\nGroups {gid_a} & {gid_b}:  "
            f"{GROUP_NAMES[gid_a]}  |  {GROUP_NAMES[gid_b]}"
        )
        report_lines.append(f"Shared concepts: {len(shared_ids)}")
        if shared_ids:
            report_lines.append(
                f"{'Group | Concept':<50}  {'Concept ID':<12}  {'Concept Name':<30}  "
                f"{'# Images (Grp ' + str(gid_a) + ')':<24}  "
                f"{'# Images (Grp ' + str(gid_b) + ')':<24}"
            )
            report_lines.append("-" * 148)
            for cid in shared_ids:
                cname  = vocab_names[concept_match_scores[:, cid].argmax()]
                prev_a = int(group_prevalence[gid_a][cid])
                prev_b = int(group_prevalence[gid_b][cid])
                line_a = f"Grp{gid_a} ({GROUP_NAMES[gid_a]}) | {cname}/#{cid}"
                line_b = f"Grp{gid_b} ({GROUP_NAMES[gid_b]}) | {cname}/#{cid}"
                report_lines.append(
                    f"{line_a:<50}  {cid:<12}  {cname:<30}  {prev_a:<24}  {prev_b:<24}"
                )
                report_lines.append(f"{line_b:<50}")
                report_lines.append("")

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "shared_concepts_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"Shared concept report saved to {report_path}")

    # ── Misaligned-vs-aligned concept image grid (test set) ──────────────────
    plot_misaligned_concept_images(
        results=results,
        group_ids=group_ids,
        group_top_sets=group_top_sets,
        group_prevalence=group_prevalence,
        concept_match_scores=concept_match_scores,
        vocab_names=vocab_names,
        sae_dir=sae_dir,
        clip_ft=clip_ft,
        device=device,
    )

    # ── Misaligned (test) vs fine-tune set concept image grid ─────────────────
    if ft_results is not None:
        plot_misaligned_vs_fttrain_concept_images(
            te_results=results,
            ft_results=ft_results,
            group_ids_te=group_ids,
            group_top_sets=group_top_sets,
            group_prevalence=group_prevalence,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            sae_dir=sae_dir,
            clip_ft=clip_ft,
            device=device,
        )


def plot_concept_top_images(
    cid, cname, group_label,
    top10_te_indices, top10_ft_indices,
    te_paths, ft_paths,
    te_reps_np, ft_reps_np,
    test_save_dir, ft_save_dir,
    te_row_title="Test (misclassified)",
    ft_row_title="FT set",
    te_labels=None,
    ft_labels=None,
):
    """
    Saves two separate plots — one per split — into their respective folders:
      test_save_dir/concept_{cid}_{cname}.png  (misclassified test images)
      ft_save_dir/concept_{cid}_{cname}.png    (ft images)
    Each plot is a 1-row × 10-col grid sorted by concept activation descending.

    te_labels / ft_labels: optional list of per-image label strings (same length
      as top10_te_indices / top10_ft_indices). When provided, each image subtitle
      becomes "<label>\\n<activation>".
    """
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL

    plot_paths = []
    for indices, paths_list, reps, row_title, save_dir, img_labels in [
        (top10_te_indices, te_paths, te_reps_np, te_row_title, test_save_dir, te_labels),
        (top10_ft_indices, ft_paths, ft_reps_np, ft_row_title, ft_save_dir, ft_labels),
    ]:
        n_cols = max(len(indices), 1)
        fig, axes = plt.subplots(1, n_cols, figsize=(2.2 * n_cols, 3))
        if n_cols == 1:
            axes = [axes]
        fig.suptitle(
            f"Concept #{cid} — {cname} | {row_title}\n{group_label}",
            fontsize=10, fontweight="bold",
        )
        for col in range(n_cols):
            ax = axes[col]
            ax.axis("off")
            if col < len(indices):
                idx = indices[col]
                act = reps[idx, cid]
                ax.imshow(_PIL.open(paths_list[idx]).convert("RGB"))
                lbl = img_labels[col] if img_labels is not None and col < len(img_labels) else ""
                title = f"{lbl}\n{act:.3f}" if lbl else f"{act:.3f}"
                ax.set_title(title, fontsize=7, pad=2)
        plt.tight_layout()
        plot_path = os.path.join(save_dir, f"concept_{cid}_{cname}.png")
        plt.savefig(plot_path, bbox_inches="tight", dpi=120)
        plt.close(fig)
        plot_paths.append(plot_path)

    return plot_paths


def analyze_misclassified_concepts(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    activation_threshold=None,
    prevalence_threshold=0.90,
    top_n_concepts=20,
):
    """
    Standalone experiment: for each misaligned group (1 = landbird/water,
    2 = waterbird/land) in the test set, identify which images are misclassified
    by the fine-tuned CLIP model, then find SAE concepts that fire above threshold
    in more than `prevalence_threshold` of those misclassified images.

    Outputs
    -------
    Report  : sae_dir/misclassified_concepts_report.txt
    Images  : sae_dir/test_images/{group_folder}/{cid}-{cname}/   ← misclassified test images
              sae_dir/ft_images/{class_folder}/{cid}-{cname}/     ← ft-set images activating concept
    """
    import shutil

    te_reps    = te_results["sae_representations"]   # (N_te, L)
    te_paths   = te_results["image_paths"]
    ft_reps    = ft_results["sae_representations"]   # (N_ft, L)
    ft_paths   = ft_results["image_paths"]
    te_reps_np = te_reps.cpu().numpy()
    ft_reps_np = ft_reps.cpu().numpy()

    if activation_threshold is None:
        activation_threshold = float(te_reps_np.mean())
        print(f"  activation_threshold (mean): {activation_threshold:.6f}")

    group_ids_arr = np.array([s[3] for s in test_ds.samples])

    # Test-set misaligned groups: 2 = landbird/water bg, 3 = waterbird/land bg
    # FT set contains only aligned groups (landbird/land and waterbird/water)
    MISALIGNED_GROUP_NAMES = {
        1: "landbird / water bg",
        2: "waterbird / land bg",
    }
    GROUP_FOLDER     = {1: "land_bird_on_water", 2: "water_bird_on_land"}
    FT_CLASS_FOLDER  = {1: "water_birds",         2: "land_birds"}

    report_lines = [
        "Misclassified Image Concept Analysis",
        f"Threshold: activation > {activation_threshold:.6f}  |  "
        f"Prevalence > {prevalence_threshold * 100:.0f}%  |  Top {top_n_concepts} concepts",
        "=" * 80,
    ]
    




    all_concepts           = []
    concept_source_group   = {}   # cid → gid_mis that contributed it
    concept_mis_prevalence = {}   # cid → prevalence in that misaligned group
    for gid_mis in [1, 2]:
        g_indices   = np.where(group_ids_arr == gid_mis)[0]

        # ── Build a group-filtered dataset with the same structure as test_ds ──
        group_ds = ManifestDataset.__new__(ManifestDataset)
        group_ds.samples         = [test_ds.samples[i] for i in g_indices]
        group_ds.clip_preprocess = clip_ft.preprocess
        group_ds.manifest_path   = test_ds.manifest_path

        # ── Classify all images in the group via clip_ft.run() ──────────────
        print(f"\nClassifying Group {gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — {len(g_indices)} images …")
        zs_stats      = clip_ft.run(dataset=group_ds, prompt_mode="shape",
                                    dataset_name="waterbirds")
        mis_local     = np.where(
            np.array(zs_stats["predictions_shape"]) != np.array(zs_stats["true_labels"])
        )[0]
        misclassified = g_indices[mis_local].tolist()
        n_mis         = len(misclassified)
        print(f"  Misclassified: {n_mis} / {len(g_indices)}")

        report_lines.append(
            f"\n{'─'*80}\n"
            f"Group {gid_mis} — {MISALIGNED_GROUP_NAMES[gid_mis]}\n"
            f"Misclassified: {n_mis} / {len(g_indices)}"
        )

        if n_mis == 0:
            report_lines.append("  No misclassified images — skipping.")
            continue

         

        # ── Concept prevalence among misclassified images ────────────────────
        mis_reps = te_reps_np[misclassified]                          # (n_mis, L)
        prev     = (mis_reps > activation_threshold).sum(axis=0) / n_mis  # (L,) fraction

        eligible = np.where(prev > prevalence_threshold)[0]
        if len(eligible) == 0:
            report_lines.append(
                f"  No concepts exceed {prevalence_threshold*100:.0f}% prevalence — skipping."
            )
            continue

        top_concepts = eligible[np.argsort(prev[eligible])[::-1]][:top_n_concepts]
        all_concepts.extend(top_concepts.tolist())
        for cid in top_concepts.tolist():
            concept_source_group[cid]   = gid_mis
            concept_mis_prevalence[cid] = float(prev[cid])
        # ── Report section for this group ────────────────────────────────────
        report_lines.append(
            f"Concepts with prevalence > {prevalence_threshold*100:.0f}%: "
            f"{len(eligible)} found, top {top_n_concepts} shown\n"
        )
        report_lines.append(
            f"{'Concept ID':<12}  {'Concept Name':<30}  {'Prevalence':>12}  {'# Images':>10}"
        )
        report_lines.append("-" * 70)
        for cid in top_concepts:
            cname   = vocab_names[concept_match_scores[:, cid].argmax()]
            n_act   = int((mis_reps[:, cid] > activation_threshold).sum())
            report_lines.append(
                f"{cid:<12}  {cname:<30}  {prev[cid]*100:>11.1f}%  {n_act:>10}"
            )

        # ── Save images per concept ──────────────────────────────────────────
        group_folder    = GROUP_FOLDER[gid_mis]
        ft_class_folder = FT_CLASS_FOLDER[gid_mis]

        for cid in top_concepts:
            cname              = vocab_names[concept_match_scores[:, cid].argmax()]

            # test_images/{group_folder}/{cid}-{cname}/
            test_out = os.path.join(sae_dir, "test_images", group_folder)
            os.makedirs(test_out, exist_ok=True)
            active_mis = sorted(
                [idx for idx in misclassified if te_reps_np[idx, cid] > activation_threshold],
                key=lambda idx: te_reps_np[idx, cid], reverse=True,
            )
            for idx in active_mis[:10]:
                shutil.copy2(te_paths[idx],
                             os.path.join(test_out, os.path.basename(te_paths[idx])))

            # ft_images/{ft_class_folder}/{cid}-{cname}/
            ft_out = os.path.join(sae_dir, "ft_images", ft_class_folder)
            os.makedirs(ft_out, exist_ok=True)
            active_ft = sorted(
                [i for i, _ in enumerate(ft_paths) if ft_reps_np[i, cid] > activation_threshold],
                key=lambda i: ft_reps_np[i, cid], reverse=True,
            )
            for i in active_ft[:10]:
                shutil.copy2(ft_paths[i], os.path.join(ft_out, os.path.basename(ft_paths[i])))

            plot_path = plot_concept_top_images(
                cid=cid, cname=cname,
                group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]})",
                top10_te_indices=active_mis[:10],
                top10_ft_indices=active_ft[:10],
                te_paths=te_paths, ft_paths=ft_paths,
                te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
                test_save_dir=test_out, ft_save_dir=ft_out,
            )
            print(f"  Concept {cid} [{cname}]: "
                  f"test→{test_out}  ft→{ft_out}  plot→{plot_path}")

            # ── Lowest-activation images — saved by each image's actual group ──
            _te_grp_folders = {
                0: "landbird_land_bg", 1: "landbird_water_bg",
                2: "waterbird_land_bg", 3: "waterbird_water_bg",
            }
            _ft_lbl_folders = {0: "land_birds", 1: "water_birds"}

            bottom_te = np.argsort(te_reps_np[:, cid])[:10]
            bottom_ft = np.argsort(ft_reps_np[:, cid])[:10]

            for idx in bottom_te:
                actual_gid    = test_ds.samples[idx][3]
                actual_folder = _te_grp_folders.get(actual_gid, f"group_{actual_gid}")
                dest = os.path.join(sae_dir, "test_images", "lowest", actual_folder)
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(te_paths[idx],
                             os.path.join(dest, os.path.basename(te_paths[idx])))

            for i in bottom_ft:
                actual_label  = ft_ds.samples[i][1]
                actual_folder = _ft_lbl_folders.get(actual_label, f"label_{actual_label}")
                dest = os.path.join(sae_dir, "ft_images", "lowest", actual_folder)
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(ft_paths[i],
                             os.path.join(dest, os.path.basename(ft_paths[i])))

            te_labels_low = [
                _te_grp_folders.get(test_ds.samples[idx][3], f"grp{test_ds.samples[idx][3]}")
                for idx in bottom_te
            ]
            ft_labels_low = [
                _ft_lbl_folders.get(ft_ds.samples[i][1], f"lbl{ft_ds.samples[i][1]}")
                for i in bottom_ft
            ]

            test_plot_dir = os.path.join(sae_dir, "test_images", "lowest", "plots")
            ft_plot_dir   = os.path.join(sae_dir, "ft_images",   "lowest", "plots")
            os.makedirs(test_plot_dir, exist_ok=True)
            os.makedirs(ft_plot_dir,   exist_ok=True)
            low_plot_path = plot_concept_top_images(
                cid=cid, cname=cname,
                group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — Lowest",
                top10_te_indices=bottom_te.tolist(),
                top10_ft_indices=bottom_ft.tolist(),
                te_paths=te_paths, ft_paths=ft_paths,
                te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
                test_save_dir=test_plot_dir, ft_save_dir=ft_plot_dir,
                te_row_title="Test (all) — lowest activation",
                ft_row_title="FT set — lowest activation",
                te_labels=te_labels_low,
                ft_labels=ft_labels_low,
            )
            print(f"  Concept {cid} [{cname}] lowest: "
                  f"test→test_images/lowest/*  ft→ft_images/lowest/*  plot→{low_plot_path}")

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "misclassified_concepts_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nMisclassified concept report saved to {report_path}")
    return all_concepts, concept_source_group, concept_mis_prevalence



def find_spurious_concepts(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    activation_threshold=None,
    prevalence_threshold=0.90,
    top_n_concepts=20,
):
    """
    Standalone experiment: for each misaligned group (1 = landbird/water,
    2 = waterbird/land) in the test set, identify which images are misclassified
    by the fine-tuned CLIP model, then find SAE concepts that fire above threshold
    in more than `prevalence_threshold` of those misclassified images.

    Outputs
    -------
    Report  : sae_dir/misclassified_concepts_report.txt
    Images  : sae_dir/test_images/{group_folder}/{cid}-{cname}/   ← misclassified test images
              sae_dir/ft_images/{class_folder}/{cid}-{cname}/     ← ft-set images activating concept
    """
    import shutil

    te_reps    = te_results["sae_representations"]   # (N_te, L)
    te_paths   = te_results["image_paths"]
    ft_reps    = ft_results["sae_representations"]   # (N_ft, L)
    ft_paths   = ft_results["image_paths"]
    te_reps_np = te_reps.cpu().numpy()
    ft_reps_np = ft_reps.cpu().numpy()

    if activation_threshold is None:
        activation_threshold = float(te_reps_np.mean())
        print(f"  activation_threshold (mean): {activation_threshold:.6f}")

    group_ids_arr = np.array([s[3] for s in test_ds.samples])

    # Test-set misaligned groups: 2 = landbird/water bg, 3 = waterbird/land bg
    # FT set contains only aligned groups (landbird/land and waterbird/water)
    MISALIGNED_GROUP_NAMES = {
        1: "landbird / water bg",
        2: "waterbird / land bg",
    }
    GROUP_FOLDER     = {1: "land_bird_on_water", 2: "water_bird_on_land"}
    FT_CLASS_FOLDER  = {1: "water_birds",         2: "land_birds"}

    report_lines = [
        "Misclassified Image Concept Analysis",
        f"Threshold: activation > {activation_threshold:.6f}  |  "
        f"Prevalence > {prevalence_threshold * 100:.0f}%  |  Top {top_n_concepts} concepts",
        "=" * 80,
    ]
    





    all_candidate_concepts  = []
    concept_source_group    = {}   # cid → gid_mis that contributed it
    concept_mis_prevalence  = {}   # cid → prevalence in that misaligned group
    group_top_sets   = {}
    group_prevalence = {}
    for gid_mis in [1, 2]:
        g_indices   = np.where(group_ids_arr == gid_mis)[0]

        # ── Build a group-filtered dataset with the same structure as test_ds ──
        group_ds = ManifestDataset.__new__(ManifestDataset)
        group_ds.samples         = [test_ds.samples[i] for i in g_indices]
        group_ds.clip_preprocess = clip_ft.preprocess
        group_ds.manifest_path   = test_ds.manifest_path

        # ── Classify all images in the group via clip_ft.run() ──────────────
        print(f"\nClassifying Group {gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — {len(g_indices)} images …")
        zs_stats      = clip_ft.run(dataset=group_ds, prompt_mode="shape",
                                    dataset_name="waterbirds")
        mis_local     = np.where(
            np.array(zs_stats["predictions_shape"]) != np.array(zs_stats["true_labels"])
        )[0]
        
        correct_local = np.where(
            np.array(zs_stats["predictions_shape"]) == np.array(zs_stats["true_labels"])
        )[0]
        

        misclassified        = g_indices[mis_local].tolist()
        correctly_classified = g_indices[correct_local].tolist()
        n_mis                = len(misclassified)
        n_correct            = len(correctly_classified)
        print(f"  Misclassified: {n_mis} / {len(g_indices)}")
        print(f"  Correctly classified: {n_correct} / {len(g_indices)}")

        report_lines.append(
            f"\n{'─'*80}\n"
            f"Group {gid_mis} — {MISALIGNED_GROUP_NAMES[gid_mis]}\n"
            f"Misclassified: {n_mis} / {len(g_indices)}"
        )

        if n_mis == 0:
            report_lines.append("  No misclassified images — skipping.")
            continue



        # ── Concept prevalence among misclassified images ────────────────────
        mis_reps  = te_reps_np[misclassified]                               # (n_mis, L)
        prev      = (mis_reps > activation_threshold).sum(axis=0) / n_mis  # (L,) fraction — for filtering & diff
        mean_prev = mis_reps.mean(axis=0)                                   # (L,) mean activation — for ranking

        eligible = np.where(prev > prevalence_threshold)[0]
        if len(eligible) == 0:
            report_lines.append(
                f"  No concepts exceed {prevalence_threshold*100:.0f}% prevalence — skipping."
            )
            continue

        # rank eligible concepts by mean activation (captures both magnitude and frequency)
        pool_size    = int(len(eligible) * .3) # int(len(eligible) * .3) #len(eligible) # if len(eligible) < 3 else int(len(eligible) * .3)
        # top_concepts = eligible[np.argsort(mean_prev[eligible])[::-1]][:top_n_concepts]  # for report
        mis_pool     = set(eligible[np.argsort(mean_prev[eligible])[::-1]][:pool_size].tolist())

        # ── Aligned group definitions ────────────────────────────────────────
        # gid_mis=1 (landbird/water bg): HIGH in group 3 (waterbird/water, same bg)
        #                                LOW  in group 0 (landbird/land)
        # gid_mis=2 (waterbird/land bg): HIGH in group 0 (landbird/land, same bg)
        #                                LOW  in group 3 (waterbird/water)
        _aligned_high = {1: 3, 2: 0}
        _aligned_low  = {1: 0, 2: 3}
        high_reps = te_reps_np[group_ids_arr == _aligned_high[gid_mis]]
        low_reps  = te_reps_np[group_ids_arr == _aligned_low[gid_mis]]

        # ── Candidate concepts: top-in-misclassified ∩ low-in-aligned-low-group ──
        low_aligned_mean = low_reps.mean(axis=0)
        low_aligned_pool = set(np.argsort(low_aligned_mean)[:pool_size].tolist())

        candidate_concepts = list(mis_pool & low_aligned_pool)
        if not candidate_concepts:
            candidate_concepts = list(mis_pool)
            print("  No mis∩low_aligned intersection — using top misclassified pool.")
        candidate_concepts = sorted(candidate_concepts, key=lambda c: prev[c], reverse=True)
        all_candidate_concepts.extend(candidate_concepts)
        for cid in candidate_concepts:
            concept_source_group[cid]   = gid_mis
            concept_mis_prevalence[cid] = float(prev[cid])

        # ── Re-rank candidates by prevalence in the aligned groups ──────────

        aligned_scores = []
        for cid in candidate_concepts:
            prev_high = (high_reps[:, cid] > activation_threshold).sum() / len(high_reps)
            prev_low  = (low_reps[:, cid] > activation_threshold).sum() / len(low_reps)
            # mean_low  = float(low_reps[:, cid].mean())   # low activation value, not count
            aligned_scores.append((cid, prev_high - prev_low))

        aligned_scores.sort(key=lambda x: x[1], reverse=True)
        #### top_concepts are the most spurious concepts that are both prevalent in misclassified images and low in the aligned low-bg group
        top_concepts = [cid for cid, _ in aligned_scores[:top_n_concepts]]
        group_top_sets[gid_mis]   = set(top_concepts)
        group_prevalence[gid_mis] = {c: float(prev[c]) for c in top_concepts}

        report_lines.append(
            f"Concepts with prevalence > {prevalence_threshold*100:.0f}%: "
            f"{len(eligible)} found, top {top_n_concepts} shown\n"
        )
        report_lines.append(
            f"{'Concept ID':<12}  {'Concept Name':<30}  {'Prevalence':>12}  {'# Images':>10}"
        )
        report_lines.append("-" * 70)
        for cid in top_concepts:
            cname   = vocab_names[concept_match_scores[:, cid].argmax()]
            n_act   = int((mis_reps[:, cid] > activation_threshold).sum())
            report_lines.append(
                f"{cid:<12}  {cname:<30}  {prev[cid]*100:>11.1f}%  {n_act:>10}"
            )

        # ── Save images per concept ──────────────────────────────────────────
        group_folder    = GROUP_FOLDER[gid_mis]
        ft_class_folder = FT_CLASS_FOLDER[gid_mis]

        for cid in top_concepts:
            cname              = vocab_names[concept_match_scores[:, cid].argmax()]

            # test_images/{group_folder}/{cid}-{cname}/
            test_out = os.path.join(sae_dir, "test_images", "spurious", group_folder)
            os.makedirs(test_out, exist_ok=True)
            active_mis = sorted(
                [idx for idx in misclassified if te_reps_np[idx, cid] > activation_threshold],
                key=lambda idx: te_reps_np[idx, cid], reverse=True,
            )
            for idx in active_mis[:10]:
                shutil.copy2(te_paths[idx],
                             os.path.join(test_out, os.path.basename(te_paths[idx])))

            # ft_images/{ft_class_folder}/{cid}-{cname}/
            ft_out = os.path.join(sae_dir, "ft_images", "spurious", ft_class_folder)
            os.makedirs(ft_out, exist_ok=True)
            active_ft = sorted(
                [i for i, _ in enumerate(ft_paths) if ft_reps_np[i, cid] > activation_threshold],
                key=lambda i: ft_reps_np[i, cid], reverse=True,
            )
            for i in active_ft[:10]:
                shutil.copy2(ft_paths[i], os.path.join(ft_out, os.path.basename(ft_paths[i])))

            plot_path = plot_concept_top_images(
                cid=cid, cname=cname,
                group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]})",
                top10_te_indices=active_mis[:10],
                top10_ft_indices=active_ft[:10],
                te_paths=te_paths, ft_paths=ft_paths,
                te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
                test_save_dir=test_out, ft_save_dir=ft_out,
            )
            print(f"  Concept {cid} [{cname}]: "
                  f"test→{test_out}  ft→{ft_out}  plot→{plot_path}")

            _te_grp_folders = {
                0: "landbird_land_bg", 1: "landbird_water_bg",
                2: "waterbird_land_bg", 3: "waterbird_water_bg",
            }
            _ft_lbl_folders = {0: "land_birds", 1: "water_birds"}

            bottom_te = np.argsort(te_reps_np[:, cid])[:10]
            bottom_ft = np.argsort(ft_reps_np[:, cid])[:10]

            for idx in bottom_te:
                actual_gid    = test_ds.samples[idx][3]
                actual_folder = _te_grp_folders.get(actual_gid, f"group_{actual_gid}")
                dest = os.path.join(sae_dir, "test_images", "lowest", actual_folder)
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(te_paths[idx],
                             os.path.join(dest, os.path.basename(te_paths[idx])))

            for i in bottom_ft:
                actual_label  = ft_ds.samples[i][1]
                actual_folder = _ft_lbl_folders.get(actual_label, f"label_{actual_label}")
                dest = os.path.join(sae_dir, "ft_images", "lowest", actual_folder)
                os.makedirs(dest, exist_ok=True)
                shutil.copy2(ft_paths[i],
                             os.path.join(dest, os.path.basename(ft_paths[i])))

            te_labels_low = [
                _te_grp_folders.get(test_ds.samples[idx][3], f"grp{test_ds.samples[idx][3]}")
                for idx in bottom_te
            ]
            ft_labels_low = [
                _ft_lbl_folders.get(ft_ds.samples[i][1], f"lbl{ft_ds.samples[i][1]}")
                for i in bottom_ft
            ]

            test_plot_dir = os.path.join(sae_dir, "test_images", "lowest", "plots")
            ft_plot_dir   = os.path.join(sae_dir, "ft_images",   "lowest", "plots")
            os.makedirs(test_plot_dir, exist_ok=True)
            os.makedirs(ft_plot_dir,   exist_ok=True)
            low_plot_path = plot_concept_top_images(
                cid=cid, cname=cname,
                group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — Lowest",
                top10_te_indices=bottom_te.tolist(),
                top10_ft_indices=bottom_ft.tolist(),
                te_paths=te_paths, ft_paths=ft_paths,
                te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
                test_save_dir=test_plot_dir, ft_save_dir=ft_plot_dir,
                te_row_title="Test (all) — lowest activation",
                ft_row_title="FT set — lowest activation",
                te_labels=te_labels_low,
                ft_labels=ft_labels_low,
            )
            print(f"  Concept {cid} [{cname}] lowest: "
                  f"test→test_images/lowest/*  ft→ft_images/lowest/*  plot→{low_plot_path}")

    # ── Build aligned-group sets so intersection logic works in plot functions ─
    # pairs: gid_mis=1 ↔ gid_aln=3,  gid_mis=2 ↔ gid_aln=0
    for gid_aln, gid_mis_pair in [(3, 1), (0, 2)]:
        paired = group_top_sets.get(gid_mis_pair, set())
        group_top_sets[gid_aln] = paired
        aln_reps = te_reps_np[group_ids_arr == gid_aln]
        group_prevalence[gid_aln] = (
            {c: float((aln_reps[:, c] > activation_threshold).mean()) for c in paired}
            if len(aln_reps) else {}
        )

    device         = next(clip_ft.model.parameters()).device
    group_ids_list = [s[3] for s in test_ds.samples]

    plot_misaligned_concept_images(
        results=te_results,
        group_ids=group_ids_list,
        group_top_sets=group_top_sets,
        group_prevalence=group_prevalence,
        concept_match_scores=concept_match_scores,
        vocab_names=vocab_names,
        sae_dir=sae_dir,
        clip_ft=clip_ft,
        device=device,
        top_n_concepts=top_n_concepts,
    )
    plot_misaligned_vs_fttrain_concept_images(
        te_results=te_results,
        ft_results=ft_results,
        group_ids_te=group_ids_list,
        group_top_sets=group_top_sets,
        group_prevalence=group_prevalence,
        concept_match_scores=concept_match_scores,
        vocab_names=vocab_names,
        sae_dir=sae_dir,
        clip_ft=clip_ft,
        device=device,
        top_n_concepts=top_n_concepts,
    )

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "spurious_concepts_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSpurious concept report saved to {report_path}")
    
    return all_candidate_concepts, top_concepts, concept_source_group, concept_mis_prevalence


def make_projection_ablation_hook(dirs):
    def hook(module, input, output):
        feat = output.float()
        for d in dirs:
            feat = feat - (feat @ d).unsqueeze(-1) * d.unsqueeze(0)
        return feat.to(output.dtype)
    return hook


def make_sae_ablation_hook(sae_model, concept_idx, device):
    """
    Returns a forward hook that routes CLIP's visual output through the SAE,
    zeros out the specified latent dimensions, then decodes back to CLIP space.

    Usage — build the hook once and pass it as ablation_hook:
        hook = make_sae_ablation_hook(sae_model, candidate_concepts, device)
        ablate_spurious_concepts(..., ablation_hook=hook)
    """
    concept_idx = list(concept_idx)

    def hook(module, input, output):
        feat = output.float()                           # (B, D) CLIP features
        latents, _ = sae_model.encode(feat)             # (B, L) sparse latents
        latents = latents.clone()
        latents[:, concept_idx] = 0.0                   # zero spurious concepts
        reconstructed = sae_model.decode(latents)       # (B, D) back to CLIP space
        return reconstructed.to(output.dtype)

    return hook


def _hook_label(fn) -> str:
    """Return a human-readable name for an ablation hook callable.

    Handles both plain functions (make_projection_ablation_hook) and
    closures returned by a factory (make_sae_ablation_hook.<locals>.hook).
    """
    qn = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "unknown")
    # "make_sae_ablation_hook.<locals>.hook" → "make_sae_ablation_hook"
    return qn.split(".<locals>")[0]


@torch.no_grad()
def ablate_spurious_concepts(
    clip_ft, sae_model, test_ds,
    spurious_concept_indices,
    device, sae_dir,
    ablation_hook=make_projection_ablation_hook,
    vocab_names=None,
    concept_match_scores=None,
    concept_source_group=None,
    concept_mis_prevalence=None,
    concept_extractor_name=None,
):
    """
    Zero out `spurious_concept_indices` SAE latents for every test image,
    reconstruct CLIP features from the ablated latents, re-classify, and
    compare per-group accuracy against the original fine-tuned CLIP predictions.

    Saves: sae_dir/ablation_report_{extractor}_{hook}.txt
            sae_dir/ablation_corrected_group{1,2}.png
    """
    import clip as openai_clip
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL

    CLASS_PROMPTS = ["a photo of a landbird", "a photo of a waterbird"]
    CLASS_NAMES   = ["landbird", "waterbird"]
    tokens = openai_clip.tokenize(CLASS_PROMPTS).to(device)
    txt    = clip_ft.model.encode_text(tokens).float()
    txt    = txt / txt.norm(dim=-1, keepdim=True)

    group_ids   = np.array([s[3] for s in test_ds.samples])
    true_labels = np.array([s[1] for s in test_ds.samples])

    # ── Spurious concept directions in CLIP feature space ───────────────────
    concept_idx = list(spurious_concept_indices)
    if sae_model.model.tied:
        dirs = sae_model.model.encoder[:, concept_idx].T.float()
    else:
        dirs = sae_model.model.decoder[concept_idx].float()
    dirs = dirs.to(device) / sae_model.scaling_factor.float()
    dirs = dirs / dirs.norm(dim=-1, keepdim=True)

    # ── Original predictions (no hook) ── store full logits for delta scoring
    orig_logits = []
    for path, _, _, _ in tqdm(test_ds.samples, desc="  Original CLIP"):
        img  = _PIL.open(path).convert("RGB")
        inp  = clip_ft.preprocess(img).unsqueeze(0).to(device)
        feat = clip_ft.model.encode_image(inp).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        orig_logits.append((feat @ txt.T).squeeze(0).cpu())
    orig_logits = torch.stack(orig_logits)          # (N, 2)
    orig_preds  = orig_logits.argmax(dim=1).numpy()

    # ── Ablated predictions ───────────────────────────────────────────────────
    ablated_logits = []
    sig    = inspect.signature(ablation_hook)
    hook   = ablation_hook(dirs) if len(sig.parameters) == 1 else ablation_hook
    handle = clip_ft.model.visual.register_forward_hook(hook)
    try:
        for path, _, _, _ in tqdm(test_ds.samples, desc="  Ablated CLIP"):
            img  = _PIL.open(path).convert("RGB")
            inp  = clip_ft.preprocess(img).unsqueeze(0).to(device)
            feat = clip_ft.model.encode_image(inp).float()
            feat = feat / feat.norm(dim=-1, keepdim=True)
            ablated_logits.append((feat @ txt.T).squeeze(0).cpu())
    finally:
        handle.remove()
    ablated_logits = torch.stack(ablated_logits)    # (N, 2)
    ablated_preds  = ablated_logits.argmax(dim=1).numpy()

    # ── Report ────────────────────────────────────────────────────────────────
    concept_names = []
    if vocab_names is not None and concept_match_scores is not None:
        for cid in concept_idx:
            concept_names.append(f"{cid}:{vocab_names[concept_match_scores[:, cid].argmax()]}")
    else:
        concept_names = [str(c) for c in concept_idx]

    # ── Per-group concept breakdown ───────────────────────────────────────────
    group_concept_counts = {}
    if concept_source_group is not None:
        for cid in concept_idx:
            gid = concept_source_group.get(cid)
            if gid is not None:
                group_concept_counts[gid] = group_concept_counts.get(gid, 0) + 1

    report_lines = [
        "Spurious concept ablation",
        "=" * 80,
        f"Ablated concepts ({len(concept_idx)}): {', '.join(concept_names)}",
    ]

    if group_concept_counts:
        for gid, cnt in sorted(group_concept_counts.items()):
            report_lines.append(
                f"  Group {gid} ({GROUP_NAMES.get(gid, str(gid))}): {cnt} concept(s)"
            )

    # ── Per-concept prevalence table ─────────────────────────────────────────
    if concept_mis_prevalence is not None:
        report_lines += [
            "-" * 80,
            f"Concept prevalence in source misaligned group:",
            f"  {'Concept':<36}  {'Source Group':<10}  {'Prevalence':>10}",
            "  " + "-" * 62,
        ]
        for cid in concept_idx:
            cname = (
                vocab_names[concept_match_scores[:, cid].argmax()]
                if vocab_names is not None and concept_match_scores is not None
                else str(cid)
            )
            src_gid = concept_source_group.get(cid, "?") if concept_source_group else "?"
            pct     = concept_mis_prevalence.get(cid, float("nan")) * 100
            report_lines.append(
                f"  {f'{cid}:{cname}':<36}  {f'Group {src_gid}':<10}  {pct:>9.1f}%"
            )

    report_lines += [
        "-" * 80,
        f"{'Group':<42}  {'Orig Acc':>9}  {'Ablated Acc':>11}  {'Δ Acc':>7}",
        "-" * 74,
    ]
    for gid in sorted(set(group_ids.tolist())):
        mask        = group_ids == gid
        orig_acc    = (orig_preds[mask]    == true_labels[mask]).mean()
        ablated_acc = (ablated_preds[mask] == true_labels[mask]).mean()
        report_lines.append(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid)):<35})  "
            f"{orig_acc*100:>8.1f}%  {ablated_acc*100:>10.1f}%  "
            f"{(ablated_acc - orig_acc)*100:>+6.1f}%"
        )
    overall_orig    = (orig_preds    == true_labels).mean()
    overall_ablated = (ablated_preds == true_labels).mean()
    report_lines += [
        "-" * 74,
        f"{'Overall':<42}  {overall_orig*100:>8.1f}%  {overall_ablated*100:>10.1f}%  "
        f"{(overall_ablated - overall_orig)*100:>+6.1f}%",
    ]

    report_text = "\n".join(report_lines) + "\n"
    extractor_tag = concept_extractor_name or "unknown_extractor"
    hook_tag      = _hook_label(ablation_hook)
    report_fname  = f"ablation_report_{extractor_tag}__{hook_tag}.txt"
    report_path   = os.path.join(sae_dir, report_fname)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"Ablation report saved to {report_path}")

    # ── Corrected-images plots for misaligned groups 1 and 2 ─────────────────
    orig_logits_np    = orig_logits.numpy()
    ablated_logits_np = ablated_logits.numpy()

    for gid in [1, 2]:
        g_indices = np.where(group_ids == gid)[0]
        if len(g_indices) == 0:
            continue

        # images where prediction flipped from wrong → correct
        was_wrong   = orig_preds[g_indices]    != true_labels[g_indices]
        now_correct = ablated_preds[g_indices] == true_labels[g_indices]
        corrected   = g_indices[was_wrong & now_correct]

        if len(corrected) == 0:
            print(f"  Group {gid}: no corrected predictions — skipping plot.")
            continue

        # score = increase in logit for the true class after ablation
        true_cls_idx = true_labels[corrected]
        scores = (
            ablated_logits_np[corrected, true_cls_idx] -
            orig_logits_np[corrected, true_cls_idx]
        )
        top_k    = min(10, len(corrected))
        top_order = np.argsort(scores)[::-1][:top_k]
        top_idx   = corrected[top_order]
        top_scores = scores[top_order]

        fig, axes = plt.subplots(1, top_k, figsize=(2.5 * top_k, 3.5), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — "
            f"Top {top_k} corrected after ablation",
            fontsize=10, fontweight="bold",
        )
        for col in range(top_k):
            idx   = top_idx[col]
            score = top_scores[col]
            ax    = axes[0, col]
            ax.imshow(_PIL.open(test_ds.samples[idx][0]).convert("RGB"))
            ax.axis("off")
            true_cls  = CLASS_NAMES[true_labels[idx]]
            orig_cls  = CLASS_NAMES[orig_preds[idx]]
            ax.set_title(
                f"true:{true_cls}\norig:{orig_cls}\nΔ={score:.3f}",
                fontsize=7,
            )
        plt.tight_layout()
        fpath = os.path.join(sae_dir, f"ablation_corrected_group{gid}.png")
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved corrected plot (group {gid}): {fpath}")

    # ── Harmed-images plots for aligned groups 0 and 3 ───────────────────────
    # Images that were correct before ablation but wrong after (collateral damage).
    # Ranked by biggest drop in the true-class logit.
    for gid in [0, 3]:
        g_indices = np.where(group_ids == gid)[0]
        if len(g_indices) == 0:
            continue

        was_correct  = orig_preds[g_indices]    == true_labels[g_indices]
        now_wrong    = ablated_preds[g_indices] != true_labels[g_indices]
        harmed       = g_indices[was_correct & now_wrong]

        if len(harmed) == 0:
            print(f"  Group {gid}: no harmed predictions — skipping plot.")
            continue

        # score = drop in logit for the true class (higher = bigger drop)
        true_cls_idx = true_labels[harmed]
        scores = (
            orig_logits_np[harmed, true_cls_idx] -
            ablated_logits_np[harmed, true_cls_idx]
        )
        top_k     = min(10, len(harmed))
        top_order = np.argsort(scores)[::-1][:top_k]
        top_idx   = harmed[top_order]
        top_scores = scores[top_order]

        fig, axes = plt.subplots(1, top_k, figsize=(2.5 * top_k, 3.5), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — "
            f"Top {top_k} harmed after ablation",
            fontsize=10, fontweight="bold",
        )
        for col in range(top_k):
            idx   = top_idx[col]
            score = top_scores[col]
            ax    = axes[0, col]
            ax.imshow(_PIL.open(test_ds.samples[idx][0]).convert("RGB"))
            ax.axis("off")
            true_cls    = CLASS_NAMES[true_labels[idx]]
            ablated_cls = CLASS_NAMES[ablated_preds[idx]]
            ax.set_title(
                f"true:{true_cls}\nabl:{ablated_cls}\nΔ={-score:.3f}",
                fontsize=7,
            )
        plt.tight_layout()
        fpath = os.path.join(sae_dir, f"ablation_harmed_group{gid}.png")
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved harmed plot (group {gid}): {fpath}")


def save_representations(results, split_name, run_dir):
    """Save representation tensors and per-image metrics to the run folder."""
    out = os.path.join(run_dir, f"representations_{split_name}")
    os.makedirs(out, exist_ok=True)

    torch.save(results["clip_representations"],
               os.path.join(out, "clip_representations.pt"))
    torch.save(results["sae_representations"],
               os.path.join(out, "sae_representations.pt"))
    torch.save(results["sae_reconstructed"],
               os.path.join(out, "sae_reconstructed.pt"))

    pd.DataFrame(results["metrics"],
                 index=results["image_paths"]).to_csv(
        os.path.join(out, "metrics.csv"), index_label="img_path")

    print(f"  Saved representations ({split_name}): {out}")
    return out


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    if args.output_dir is None:
        args.output_dir = os.path.join(".", "results", args.dataset)

    # ── Resolve run folder + load metadata from model.pt ─────────────────────
    run_dir, metadata = resolve_run_dir(args)
    # run_dir = os.path.dirname(run_dir)
    print(f"\nRun folder : {os.path.abspath(run_dir)}")
    print(f"Metadata   : {metadata}")

    # ── Load datasets from exact manifests ────────────────────────────────────
    train_ds = ManifestDataset(os.path.join(run_dir, "train_all_manifest.csv"))
    ft_ds    = ManifestDataset(os.path.join(run_dir, "ft_train_manifest.csv"))
    test_ds  = ManifestDataset(os.path.join(run_dir, "test_manifest.csv"))

    print(f"\nDatasets (from saved manifests):")
    print(f"  train_ds : {len(train_ds):5d} images")
    print(f"  ft_ds    : {len(ft_ds):5d} images  ← used for fine-tuning")
    print(f"  test_ds  : {len(test_ds):5d} images")

    # ── Load fine-tuned CLIP model ────────────────────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    clip_ft = CLIPZeroShot.load_model(os.path.join(run_dir, "model.pt"), device=device)
    clip_ft.model.eval()
    print(f"\nFine-tuned model loaded  (device={device})")

    # ── Load zero-shot (base) CLIP model ──────────────────────────────────────
    # clip_zs = CLIPZeroShot(model_name=args.clip_model, device=device)
    # clip_zs.model.eval()
    # print(f"Zero-shot model loaded   ({args.clip_model})")


    

    # ── SAE representation extraction (optional) ──────────────────────────────
    from sae import SAE  # type: ignore[import]

    
    # resolve ft feature file and embed dim from previously saved .npy in run_dir
    embedding_path = os.path.join(os.path.dirname(run_dir), "embeddings")
    model_tag = args.clip_model.replace("/", "~")
    ft_npy_files = glob.glob(os.path.join(embedding_path, f"{args.dataset}_{model_tag}_ft*_train_*.npy"))
    if not ft_npy_files:
        print(f"\nNo ft feature file found in {run_dir}. Run feature extraction first.")
        return
    ########### UPDATE TO CHOOSE CORRECT FILE BASED ON FINETUNING TAG (BIASED vs FULL) AND MAX_SAMPLES
    ft_feat_path = max(ft_npy_files, key=os.path.getmtime)
    d = int(os.path.basename(ft_feat_path).replace(".npy", "").split("_")[-1])
    print(f"\nFT feature file : {ft_feat_path}  (embed_dim={d})")

    # ── Build activation string exactly as train.py does ────────────────────────
    _act = args.activation
    if args.model == "ReLUSAE" and "_" in _act:
        _act_base, _sparse_w = _act.split("_", 1)
        activation_str = f"{_act_base}_{str(float(f'0.{_sparse_w}')).split('.')[1]}"
    elif args.model in ["MSAE_UW", "MSAE_RW"]:
        model = args.model.replace("MSAE_", "")
        activation_str = "TopK_64" + "_" + model
    
    else:
        activation_str = _act
    
    sae_tag = f"{args.model}_{activation_str}"

    if args.sae_path is not None:
        # explicit file or directory override
        if os.path.isfile(args.sae_path):
            candidates = [args.sae_path]
        else:
            candidates = glob.glob(os.path.join(args.sae_path, "*.pth"))
    else:
        # auto-locate: results/<dataset>_<model>_train_image_*/sae_weights/*.pth
        model_tag = args.clip_model.replace("/", "~")
        sae_weights_pattern = os.path.join(
            "results",
            f"{args.dataset}_{model_tag}_fttrain_image_*",
            "sae_weights", "*.pth",
        )
        candidates = glob.glob(sae_weights_pattern)

    if not candidates:
        print(f"\nNo SAE weights found. Skipping SAE extraction.")
        return

    # keep only files whose filename encodes n_inputs=d AND the activation string
    consistent = [
        p for p in candidates
        if f"_{d}_" in os.path.basename(p) and f"_{activation_str}_" in os.path.basename(p)
    ]
    if not consistent:
        print(
            f"\nNo SAE weights with n_inputs={d} and activation={activation_str} found among:\n"
            + "\n".join(f"  {p}" for p in candidates)
            + "\nSkipping SAE extraction."
        )
        return

    if len(consistent) > 1:
        print(f"  Found {len(consistent)} matching SAE weights — using most recent.")
    sae_path  = max(consistent, key=os.path.getmtime)
    print(f"\nLoading SAE from: {sae_path}")
    sae_model = SAE(sae_path).to(device).eval()
    print(f"  n_inputs={sae_model.input_dim}  n_latents={sae_model.latent_dim}")

    sae_dir = os.path.join(run_dir, sae_tag)
    os.makedirs(sae_dir, exist_ok=True)
    print(f"  Output folder: {sae_dir}")
    

    

    print("\nExtracting SAE representations — ft_train...")
    ft_results_path = os.path.join(sae_dir, "representations_ft_train")
    if os.path.exists(ft_results_path):
        print(f"  Found existing representations at {ft_results_path} — loading.")
        _m = pd.read_csv(os.path.join(ft_results_path, "metrics.csv"), index_col="img_path")
        ft_results = {
            "clip_representations": torch.load(os.path.join(ft_results_path, "clip_representations.pt"),
                                               map_location="cpu", weights_only=False),
            "sae_representations":  torch.load(os.path.join(ft_results_path, "sae_representations.pt"),
                                               map_location="cpu", weights_only=False),
            "sae_reconstructed":    torch.load(os.path.join(ft_results_path, "sae_reconstructed.pt"),
                                               map_location="cpu", weights_only=False),
            "metrics":     _m.to_dict(orient="list"),
            "image_paths": _m.index.tolist(),
        }
    else:
        ft_results = extract_sae_representations(clip_ft, sae_model, ft_ds, device)
        save_representations(ft_results, "ft_train", sae_dir)

    
    print("\nExtracting SAE representations — test...")
    te_results_path = os.path.join(sae_dir, "representations_test")
    if os.path.exists(te_results_path):
        print(f"  Found existing representations at {te_results_path} — loading.")
        _m = pd.read_csv(os.path.join(te_results_path, "metrics.csv"), index_col="img_path")
        te_results = {
            "clip_representations": torch.load(os.path.join(te_results_path, "clip_representations.pt"),
                                               map_location="cpu", weights_only=False),
            "sae_representations":  torch.load(os.path.join(te_results_path, "sae_representations.pt"),
                                               map_location="cpu", weights_only=False),
            "sae_reconstructed":    torch.load(os.path.join(te_results_path, "sae_reconstructed.pt"),
                                               map_location="cpu", weights_only=False),
            "metrics":     _m.to_dict(orient="list"),
            "image_paths": _m.index.tolist(),
        }
    else:
    
        te_results = extract_sae_representations(clip_ft, sae_model, test_ds, device)
        save_representations(te_results, "test", sae_dir)

    #### plot outputs
    # import matplotlib.pyplot as plt
    # # plot histogram of activations
    # plt.figure(figsize=(10, 3))
    # plt.title("Flattened Activation Frequency-SAE Representations--ft_train")
    # plt.yscale('log')
    # plt.hist(ft_results["sae_representations"].flatten().cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "activation_histogram_ft_train.png"))


    # not needed
    # mean_activations = ft_results["sae_representations"].mean(dim=0)
    # std_activations = ft_results["sae_representations"].std(dim=0)

    # plt.figure(figsize=(10, 3))
    # plt.title("Mean Activation Frequency _ft")
    # plt.yscale('log')
    # plt.hist(mean_activations.cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "mean_activation_histogram_ft_train.png"))
    
    
    # plt.figure(figsize=(10, 3))
    # plt.title("Flattened Centered Activation Frequency_ft_train")
    # plt.yscale('log')
    # plt.hist(torch.nn.functional.relu(ft_results["sae_representations"]-mean_activations).flatten().cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "centered_activation_histogram_ft_train.png"))

    # ########## test set results
    # plt.figure(figsize=(10, 3))
    # plt.title("Flattened Activation Frequency-SAE Representations--ft_test")
    # plt.yscale('log')
    # plt.hist(te_results["sae_representations"].flatten().cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "activation_histogram_ft_test.png"))

    # mean_activations = te_results["sae_representations"].mean(dim=0)
    # std_activations = te_results["sae_representations"].std(dim=0)

    # plt.figure(figsize=(10, 3))
    # plt.title("Mean Activation Frequency _test")
    # plt.yscale('log')
    # plt.hist(mean_activations.cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "mean_activation_histogram_ft_test.png"))
    
    
    # plt.figure(figsize=(10, 3))
    # plt.title("Flattened Centered Activation Frequency_ft_test")
    # plt.yscale('log')
    # plt.hist(torch.nn.functional.relu(te_results["sae_representations"]-mean_activations).flatten().cpu().numpy(), bins=30)
    # plt.xlim(0.0, 2)
    # # plt.show()
    # plt.savefig(os.path.join(sae_dir, "centered_activation_histogram_ft_test.png"))

    ##### extract the names of concepts from vocab and show these names beside the top 10 concepts for the image with the highest activation in the test set
    
    if args.model in ["ReLUSAE", "TopKSAE", "BatchTopKSAE"]:
        vocab_path = os.path.join(embedding_path, f"concept_match/{args.model}/{args.activation}")
    else:   ###MSAE_RW or MSAE_UW
        vocab_path = os.path.join(embedding_path, f"concept_match/{args.model}")
    
    # vocab_path may point to a directory containing one or more .npy files
    if os.path.isdir(vocab_path):
        npy_files = sorted(glob.glob(os.path.join(vocab_path, "*.npy")))
        if not npy_files:
            raise FileNotFoundError(f"No .npy files found in directory {vocab_path}")
        vocab_path_to_load = npy_files[0]
    else:
        vocab_path_to_load = vocab_path
    concept_match_scores = np.load(vocab_path_to_load)
    print("concept_match_scores.shape:", concept_match_scores.shape)
    
    with open('msae/vocab/clip_disect_20k.txt', 'r') as f:
        vocab_names = [line.strip() for line in f.readlines()]
    print("Number of concept names:", len(vocab_names))
    
    
    
    ################# Concept detection
    # top1 = find_top_concept_images(
    #     te_results, top_k=1,
    #     mean_activations=mean_activations,
    #     concept_match_scores=concept_match_scores,
    #     vocab_names=vocab_names,
    #     save_path=os.path.join(sae_dir, "concept_detection_example.png"),
    # )[0]
    # image_index  = top1["image_index"]
    # argmax_index = top1["concept_index"]
    # print(f'argmax index: {argmax_index}, value: {top1["activation"]}')
    # ##### concept translation
    
    # top5_names = torch.from_numpy(concept_match_scores)[:,image_index].topk(5)
    # for i in range(5):
    #     print(f'Top 5 concept for image {image_index}: {vocab_names[top5_names.indices[i]]} with score {top5_names.values[i].item():.2f}')
    # ##### get the highest magnitude concept for the image from top 10 and find images that activate it the most
    # from PIL import Image
    # highest_concept_index = argmax_index
    # print(f'Highest magnitude concept index: {highest_concept_index}')
    # concept_activations = te_results["sae_representations"][:,highest_concept_index] - mean_activations[highest_concept_index]
    # top_images = torch.topk(concept_activations, 5)
    # fig, ax = plt.subplots(1, 5, figsize=(15, 3))
    # for i, idx in enumerate(top_images.indices.cpu().numpy()):
    #     ax[i].imshow(Image.open(te_results["image_paths"][idx]).convert('RGB'))
    #     ax[i].axis('off')
    #     ax[i].set_title(f'Image {idx}\nActivation: {concept_activations[idx].item():.2f}')
    # plt.tight_layout()
    # plt.savefig(os.path.join(sae_dir, "top_concept_images.png"))
    # plt.show()

    

    ##### Group concept distribution + shared-concept report + misaligned image grids
    ##### Experiments set 2 
    # group_ids = [s[3] for s in test_ds.samples]
    # analyze_group_concepts(
    #     results=te_results,
    #     group_ids=group_ids,
    #     sae_model=sae_model,
    #     concept_match_scores=concept_match_scores,
    #     vocab_names=vocab_names,
    #     sae_dir=sae_dir,
    #     clip_ft=clip_ft,
    #     device=device,
    #     ft_results=ft_results,
    # )

    ##### Misclassified-image concept analysis 
    ####### experiment set 3: for each misaligned group, find concepts that are highly prevalent among the misclassified images, then show the top activating test images for those concepts alongside the top activating fine-tune set images for the same concepts
    # candidate_concepts, concept_source_group, concept_mis_prevalence = analyze_misclassified_concepts(
    #     te_results=te_results,
    #     ft_results=ft_results,
    #     test_ds=test_ds,
    #     ft_ds=ft_ds,
    #     concept_match_scores=concept_match_scores,
    #     vocab_names=vocab_names,
    #     sae_dir=sae_dir,
    #     clip_ft=clip_ft,
    # )


    ### Experiment set 4: find concepts that are highly prevalent among the misclassified images and have low prevalence among the fine-tune set images, as potential spurious correlates learned by the model
    candidate_concepts, top_concepts, concept_source_group, concept_mis_prevalence = find_spurious_concepts(
        te_results=te_results,
        ft_results=ft_results,
        test_ds=test_ds,
        ft_ds=ft_ds,
        concept_match_scores=concept_match_scores,
        vocab_names=vocab_names,
        sae_dir=sae_dir,
        clip_ft=clip_ft,
    )

    if len(candidate_concepts) > 0:
        ablate_spurious_concepts(
            clip_ft=clip_ft,
            sae_model=sae_model,
            test_ds=test_ds,
            spurious_concept_indices=candidate_concepts,
            device=device,
            sae_dir=sae_dir,
            vocab_names=vocab_names,
            concept_match_scores=concept_match_scores,
            ablation_hook=make_sae_ablation_hook(sae_model, candidate_concepts, device), # make_projection_ablation_hook,     ###make_sae_ablation_hook(sae_model, candidate_concepts, device),
            concept_source_group=concept_source_group,
            concept_mis_prevalence=concept_mis_prevalence,
            concept_extractor_name="find_spurious_concepts",
        )

if __name__ == "__main__":
    main()
