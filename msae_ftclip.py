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
from ast import arg
import glob
import os
import re
import sys
import inspect  ## for context_decorator for ablation hooks
from einops import rearrange
from torchvision import transforms

from msae import sae
from clip_lrp import CLIPLRPWrapper
import numpy as np
import cv2
import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor
from clip_lrp import CLIPLRPWrapper


_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _PROJECT_ROOT)
sys.path.insert(0, os.path.join(_PROJECT_ROOT, "msae"))

import metrics as msae_metrics  # type: ignore[import]
from clip_zero_shot import CLIPZeroShot

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
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
    parser.add_argument("--clip_mode",   type=str, default="finetuned",
                        choices=["zeroshot", "finetuned"],
                        help="Whether to load the fine-tuned checkpoint (--run_dir/model.pt) "
                             "or run base CLIP zero-shot.")
    parser.add_argument("--hf_model_dir", type=str, default=None,
                        help="Local directory containing pre-downloaded HF models "
                             "(e.g. ./hf_models). Use on servers without internet. "
                             "Expected layout: <hf_model_dir>/openai/clip-vit-base-patch32/")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Results directory (default: ./results/<dataset>)")
    parser.add_argument("--batch_size",  type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--ft_epochs",   type=int, default=3)
    parser.add_argument("--ft_lr",       type=float, default=1e-5)
    parser.add_argument("--max_samples", type=int, default=400)
    parser.add_argument("--biased_ft",   type=bool, default=False)
    # ── optional explicit override ────────────────────────────────────────────
    parser.add_argument("--run_dir",     type=str, default="results/waterbirds/clip_ft_BIASED_400_20260709_111654",
                        help="Explicit path to a clip_ft_*/ run folder. "
                             "If omitted, matched from the other args via model metadata.")
    parser.add_argument("--sae_path",    type=str, default="routesae_weights/routesae_K32_clip_ft_BIASED_400_20260709_111654_16384.pt",
                        help="Path to a trained SAE checkpoint: an msae/train.py .pth file for "
                             "MSAE/ReLUSAE/TopKSAE, or a routesae_weights/*.pt file for RouteSAE "
                             "(default here is the fine-tuned RouteSAE K=32 checkpoint matching "
                             "--run_dir's default -- pass one explicitly for MSAE, a different "
                             "K, or a different --run_dir).")
    # parser.add_argument("--sae_path",    type=str, default="results/waterbirds_ViT-B~32_ft400_fttrain_image_762_512/sae_weights/4096_512_TopKReLU_256_RW_False_False_0.0_waterbirds_ViT-B~32_ft400_fttrain_image_762_512.pth",
    #                     help="Path to a trained SAE .pth file (msae/train.py output). "
    #                          "If provided, representations are extracted after feature extraction.")
    # ── SAE architecture — mirrors msae/train.py ─────────────────────────────
    parser.add_argument("-m", "--model",      type=str, default="RouteSAE",
                        choices=["ReLUSAE", "TopKSAE", "BatchTopKSAE", "MSAE_UW", "MSAE_RW", "RouteSAE"],
                        help="SAE model architecture (matches msae/train.py --model). "
                             "RouteSAE is a different architecture entirely (per-layer/per-patch "
                             "residual-stream SAE, not msae/train.py's single-embedding family) -- "
                             "requires --sae_path pointing directly at a routesae_weights/*.pt "
                             "checkpoint (no auto-discovery). Both --editing_method deactivation "
                             "and projection are supported (see routesae.py's "
                             "RouteHookProjection/hook_routesae_projection).")
    parser.add_argument("--routesae_k", type=int, default=32,
                        help="RouteSAE's top-k sparsity (--model RouteSAE only). Must match the "
                             "k the checkpoint given via --sae_path was trained with -- k isn't "
                             "encoded in the checkpoint's tensor shapes, so a mismatch loads "
                             "silently but selects the wrong number of active concepts.")
    parser.add_argument("-a", "--activation", type=str, default="TopKReLU_256",
                        help="SAE activation string (matches msae/train.py --activation, "
                             "e.g. 'ReLU_03', 'TopKReLU_64')")
    parser.add_argument("--knn_k",               type=int,   default=10,
                        help="Number of neighbors for k-NN in candidate_selection.")
    parser.add_argument("--prevalence_threshold", type=float, default=0.15,
                        help="Min fraction of candidate images a concept must influence "
                             "to be selected as a candidate concept (0–1).")
    parser.add_argument("--n_cpu_workers", type=int, default=10,
                        help="Number of CPU workers for parallel concept counting. "
                             "Defaults to os.cpu_count().")
    parser.add_argument("--concept_pool_batch_size", type=int, default=64,
                        help="Batch size for labelfree's causal concept search "
                             "(Generate_Concept_Pool). Purely a throughput knob: it groups the "
                             "same (concept, image) pairs into fewer, larger CLIP forward passes "
                             "and does not change which concepts are found. labelfree is by far "
                             "the slowest concept-finding method (~6M pairs on RouteSAE K=32, "
                             "~94k passes at 64), so this is the main lever on its runtime. "
                             "64 suits laptop/MPS; on an 80GB H100 try 256-512. RouteSAE's "
                             "ceiling is lower than MSAE's -- it holds a "
                             "(B, n_patches, latent_size) tensor here, ~50x the footprint.")
    parser.add_argument("--top_k_concepts", type=int, default=20,
                        help="If set, select the top-K concepts by count instead of "
                             "using --prevalence_threshold. Mutually exclusive with threshold mode.")
    parser.add_argument("--denoise_percentile", type=float, default=0.8,
                        help="Percentile for denoising candidate concepts (0–1).")
    parser.add_argument("--denoise_beta", type=float, default=0.8,
                        help="Beta parameter for denoising (0–1).")
    parser.add_argument("--concept_finding_method", type=str, default="highmag",
                        choices=["labelfree", "labelguided", "dialguided", "prism_baseline", "highmag", "none"],
                        help="Method for finding candidate concepts: "
                             "'labelfree' uses the SAE to find high-magnitude concepts; "
                             "'labelguided' uses the label to find concepts correlated with it. (our implementation to use misaligned misclassified examples)"
                             "'dialguided' uses the zeroshot clip label to find concepts correlated with it. (following DIAL method of label-free paper) "
                             " 'prism_baseline' does not find concepts, uses text embedings only and remove their projection "
                             "'none' finds zero concepts -- use to get predictions purely from SAE "
                             "reconstruction (encode then decode, nothing zeroed), no concept removal "
                             "at all. Reuses the ablation report machinery with an empty concept list." )

    parser.add_argument("--denoise_concepts", action="store_true", default=True,
                        help="If set, denoise candidate concepts by removing those "
                             "who are far from the mean activation.")
    parser.add_argument("--editing_method", type=str, default="projection",
                        choices=["deactivation", "projection","prism_baseline"],
                        help="Method for editing concepts: "
                             "'deactivation' zeroes out the concept; "
                             "'projection' projects the representation onto the orthogonal complement."
                             "'prism_baseline' does not edit concepts, uses text embedings only and remove their projection " )
    parser.add_argument("--projection_method", type=str, default="qr",
                        choices=["qr", "pinv"],
                        help="How to build the projection matrix for --editing_method projection "
                             "(ignored otherwise): "
                             "'qr' orthonormalizes the concept decoder directions via QR, then "
                             "projects with Q @ Q.T (matches the method used in <paper>); "
                             "'pinv' builds the projection directly as W.T @ pinv(W @ W.T) @ W, "
                             "W = the (weighted) concept decoder directions -- the standard "
                             "closed-form projector onto span(W) without orthonormalizing first. "
                             "Both project onto the same subspace when W has full row rank, so "
                             "they should agree numerically; pinv (via SVD) also degrades "
                             "gracefully when concept directions are near-duplicate/near-parallel, "
                             "which QR does not.")

    parser.add_argument("--ablation_coefficient", type=float, default=.9,
                        help="Coefficient for the ablation hook. - lambda in the paper. ")
    parser.add_argument("--skip_ablation", action="store_true",
                        help="Skip the ablation step (and its report), running only concept "
                             "finding + MACO visualization. For sharding MACO across a SLURM "
                             "array: have one 'primary' task run ablation+MACO normally, and "
                             "additional shard tasks pass --skip_ablation --maco_shard i/N to "
                             "parallelize MACO without redundantly redoing ablation per shard.")
    # ______ MACO parameters ______
    parser.add_argument("--maco_visualize_concepts", action="store_true", default=False,
                        help="Synthesize one MACO visualization per candidate concept before "
                             "ablation. OFF by default: it's an expensive opt-in stage (one "
                             "optimization loop per concept, and concept-finding methods like "
                             "labelfree/dialguided can return hundreds), so runs that only want "
                             "ablation numbers shouldn't pay for it. Previously this defaulted to "
                             "True but took a VALUE rather than being a flag, so "
                             "'--maco_visualize_concepts False' passed the truthy string 'False' "
                             "and ran anyway -- there was no way to turn it off.")
    parser.add_argument("--no_maco_visualize_concepts", dest="maco_visualize_concepts",
                        action="store_false",
                        help="Explicitly disable MACO visualization (it is already off by "
                             "default; this exists so a caller that sets it on elsewhere can "
                             "override back off).")
    parser.add_argument("--maco_max_concepts", type=int, default=200,
                        help="Cap on how many concepts get a MACO visualization. MACO runs one "
                             "optimization loop per concept, so a method returning thousands "
                             "(dialguided has hit 12132 on RouteSAE) would never finish. When "
                             "the candidate list is longer, the concepts are ranked by peak "
                             "activation and only the top N are rendered. 0 or negative "
                             "disables the cap. Only limits VISUALIZATION -- ablation always "
                             "uses the full candidate list.")
    parser.add_argument("--maco_num_steps", type=int, default=128,
                        help="MACO optimization steps per concept chunk. Runtime scales "
                             "linearly; fewer steps are faster but give less converged, "
                             "cruder visualizations. 64 is a good speed/quality tradeoff, "
                             "128 for final figures.")
    parser.add_argument("--maco_num_crops", type=int, default=16,
                        help="Random crops per step. Runtime and memory scale linearly; "
                             "more crops average out noise and give cleaner, more robust "
                             "images. 8 is fine for quick previews.")
    parser.add_argument("--maco_reference_samples", type=int, default=64,
                        help="Number of images used to estimate the MACO Fourier magnitude prior.")
    parser.add_argument("--maco_concept_batch", type=int, default=4,
                        help="Concepts optimized jointly per CLIP batch (~linear speedup, "
                             "identical results). Raises memory use: effective batch = "
                             "maco_concept_batch x maco_num_crops. Lower it if you hit OOM.")
    parser.add_argument("--maco_early_stop_patience", type=int, default=0,
                        help="Stop a chunk early after this many consecutive steps without "
                             "activation improvement (0 = always run all steps). Saves time "
                             "on quickly-converging concepts; values below ~10 risk stopping "
                             "before fine crop scales are reached, degrading image detail.")
    parser.add_argument("--maco_early_stop_delta", type=float, default=1e-3,
                        help="Minimum mean-activation improvement counted as progress by "
                             "early stopping. Larger values stop sooner (faster, rougher).")
    parser.add_argument("--maco_shard", type=str, default="0/1",
                        help="Process only every N-th candidate concept, offset i, given "
                             "\"i/N\" — for SLURM job arrays, e.g. "
                             "--maco_shard \"$SLURM_ARRAY_TASK_ID/$SLURM_ARRAY_TASK_COUNT\".")
    parser.add_argument("--maco_workers", type=int, default=10,
                        help="Run MACO for concepts in parallel worker PROCESSES (1 = "
                             "serial). On CPU (incl. Mac/MPS, where MACO runs on CPU) each "
                             "worker is capped to cpu_count//workers threads — useful "
                             "because CLIP's CPU math scales sub-linearly, so a few "
                             "thread-capped workers beat one all-core process. On a "
                             "multi-GPU node one worker is placed per GPU. On a single GPU "
                             "process parallelism gives nothing (use --maco_shard across a "
                             "SLURM array instead) and is auto-disabled. Each worker "
                             "reloads CLIP+SAE, so use >1 only when you have many concepts.")
    parser.add_argument("--concept_match_vocab", type=str, default="waterbirds_domain",
                        help="Substring selecting which concept_match .npy (i.e. which "
                             "naming vocabulary) to load, e.g. \"waterbirds_domain\" or "
                             "\"disect\". Combined with the current SAE weights basename so "
                             "the scores belong to THIS SAE. Pass \"\" to use the newest "
                             "matching file for this SAE regardless of vocabulary.")
    # ── [PRISM] orthogonal projection debiasing ───────────────────────────────
    parser.add_argument("--prism_eval", action="store_true",
                        help="[PRISM] Run zero-shot evaluation with and without "
                             "orthogonal projection debiasing after loading the CLIP model.")
    parser.add_argument("--prism_class_names", nargs="+",
                        default=["Landbird", "Waterbird"],
                        help="[PRISM] Capitalised class names for text encoding "
                             "(default matches PRISM paper: ['Landbird','Waterbird'])")
    parser.add_argument("--prism_templates", nargs="+",
                        default=["a photo of a {}.", "a picture of a {}."],
                        help="[PRISM] Template strings with {} placeholder — one embedding "
                             "per template is averaged to form the class text vector.")
    parser.add_argument("--prism_spurious_words", nargs="+",
                        default=["water", "land"],
                        help="[PRISM] Words/phrases that define the spurious subspace "
                             "to project out of image embeddings.")
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
        if args.clip_mode == "zeroshot":
            # Zero-shot loads pretrained CLIP directly (see caller), not this
            # dir's model.pt -- metadata is only ever used in a print
            # statement, so don't require a fine-tuned checkpoint to exist
            # here at all (a pure zero-shot run dir has none).
            return args.run_dir, {}
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

        # SAE.encode() → (sparse_latents, dense_latents)
        # sparse_latents: TopK-selected (exactly k non-zero for TopK/MSAE models)
        # dense_latents:  ReLU on ALL features without top-k gate — discarded here.
        # Must use sparse_latents: for TopK/MSAE, SAE.forward() returns dense_latents
        # causing L0≈0 because forward_eval bypasses the top-k gate entirely.
        sparse_latents, _ = sae_model.encode(features)
        post_reconstructed = sae_model.decode(sparse_latents)

        clip_reps.append(features.cpu().flatten())
        sae_reps.append(sparse_latents.cpu().flatten())
        sae_recons.append(post_reconstructed.cpu().flatten())
        image_paths.append(path)

        fvu = msae_metrics.explained_variance(features, post_reconstructed)
        mae = msae_metrics.normalized_mean_absolute_error(features, post_reconstructed)
        cs  = F.cosine_similarity(features, post_reconstructed)
        l0  = msae_metrics.l0_messure(sparse_latents)
        highest_magnitude = sparse_latents.max(dim=-1).values

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
        plt.close(fig)

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
    
    
    
    
    
    CLASS_PROMPTS = ["a photo of a Landbird", "a photo of a Waterbird"] ### should update to follow prism paper and use the same prompts as in prism zero-shot evaluation
    CLASS_LABELS  = ["Landbird", "Waterbird"] ### should update to Landbird and Waterbird to follow prism paper and use the same prompts as in prism zero-shot evaluation
        #### need to check if fine-tunning process and sae training process needs to update
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
        plt.close(fig)
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
        plt.close(fig)
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
    plt.close()

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
    sae_dir=None,
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
    from einops import rearrange
    from torchvision import transforms
    from overcomplete.visualization import show
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
    prevalence_threshold=0.30,
    top_n_concepts=50,
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


_HF_MODEL_NAMES = {
    "ViT-B/32": "openai/clip-vit-base-patch32",
    "ViT-B~32": "openai/clip-vit-base-patch32",
    "ViT-B/16": "openai/clip-vit-base-patch16",
    "ViT-B~16": "openai/clip-vit-base-patch16",
    "ViT-L/14": "openai/clip-vit-large-patch14",
}
_lrp_wrapper_cache: dict = {}
_hf_model_dir: str | None = None   # set from args.hf_model_dir in main(); used by all from_pretrained calls


def _hf_model_path(hf_name: str) -> str:
    """Return local path if --hf_model_dir was given, else the HuggingFace hub id."""
    if _hf_model_dir:
        return os.path.join(_hf_model_dir, hf_name)
    return hf_name

def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)  # applyColorMap outputs BGR; img is RGB
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return cam

def generate_clip_saliency(
    images: list,
    prompts: list,
    clip_ft,
    embed_modifier=None,
) -> list[dict]:
    """
    Occlusion-based patch saliency for each (image, prompt) pair.

    For each patch, the patch pixels are zeroed and the drop in cosine similarity
    is recorded as its importance score.  Uses the cached HuggingFace model from
    _lrp_wrapper_cache so the HF model is only loaded once across both methods.

    embed_modifier : optional callable (unnorm_embed) -> unnorm_embed applied after
        visual_projection, before L2-normalization — for BOTH the baseline and every
        masked forward pass. Because occlusion perturbs the INPUT and re-runs the
        full (edited) model each time, this produces a genuinely different saliency
        map for an edited model, unlike gradient/attention attribution where the
        edit sits after the attended layers.

    Returns a list of dicts with the same keys as generate_clip_heatmaps:
        "heatmap_bgr"  : np.ndarray uint8 BGR overlay
        "heatmap_rgb"  : np.ndarray uint8 RGB overlay
        "attribution"  : np.ndarray float32 [grid_size, grid_size]
        "similarity"   : float baseline cosine similarity
    """
    device = next(clip_ft.model.parameters()).device
    model_name = clip_ft.model_name

    if model_name not in _lrp_wrapper_cache:
        hf_name = _hf_model_path(_HF_MODEL_NAMES.get(model_name, "openai/clip-vit-base-patch32"))
        hf_model = CLIPModel.from_pretrained(hf_name, attn_implementation="eager").to(device)
        hf_processor = CLIPProcessor.from_pretrained(hf_name)
        _lrp_wrapper_cache[model_name] = CLIPLRPWrapper(hf_model, hf_processor)

    wrapper = _lrp_wrapper_cache[model_name]
    clip_model = wrapper.model
    clip_processor = wrapper.processor
    gs = wrapper.grid_size        # e.g. 7 for ViT-B/32, 14 for ViT-B/16
    ps = wrapper.patch_size       # e.g. 32 or 16
    img_size = wrapper.image_size # 224

    results = []
    for image, prompt in zip(images, prompts):
        clip_image = image.convert("RGB")

        image_inputs = clip_processor(images=clip_image, return_tensors="pt")
        text_inputs  = clip_processor(text=[prompt], return_tensors="pt", padding=True)

        pixel_values = image_inputs["pixel_values"].to(device)
        text_ids     = text_inputs["input_ids"].to(device)
        text_mask    = text_inputs["attention_mask"].to(device)

        with torch.no_grad():
            vision_out   = clip_model.vision_model(pixel_values=pixel_values, return_dict=True)
            img_emb      = clip_model.visual_projection(vision_out.pooler_output)
            if embed_modifier is not None:
                img_emb = embed_modifier(img_emb)
            img_emb      = img_emb / img_emb.norm(dim=-1, keepdim=True)

            text_out     = clip_model.text_model(input_ids=text_ids, attention_mask=text_mask, return_dict=True)
            txt_emb      = clip_model.text_projection(text_out.pooler_output)
            txt_emb      = txt_emb / txt_emb.norm(dim=-1, keepdim=True)

            baseline_sim = (img_emb * txt_emb).sum(dim=-1).item()

        patch_saliency_list = []
        n_patches = gs * gs
        for patch_idx in range(n_patches):
            pv_masked = pixel_values.clone().detach()
            r = patch_idx // gs
            c = patch_idx % gs
            pv_masked[:, :, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps] = 0

            with torch.no_grad():
                vo_m    = clip_model.vision_model(pixel_values=pv_masked, return_dict=True)
                ie_m    = clip_model.visual_projection(vo_m.pooler_output)
                if embed_modifier is not None:
                    ie_m = embed_modifier(ie_m)
                ie_m    = ie_m / ie_m.norm(dim=-1, keepdim=True)
                masked_sim = (ie_m * txt_emb).sum(dim=-1).item()

            patch_saliency_list.append(max(0.0, baseline_sim - masked_sim))

        patch_saliency = np.array(patch_saliency_list).reshape(gs, gs)
        if patch_saliency.max() > 0:
            patch_saliency = patch_saliency / patch_saliency.max()

        sal_t = torch.from_numpy(patch_saliency[np.newaxis, np.newaxis].astype(np.float32))
        sal_up = torch.nn.functional.interpolate(
            sal_t, size=(img_size, img_size), mode="bilinear", align_corners=False,
        ).squeeze().numpy()

        image_array = np.array(clip_image.resize((img_size, img_size))) / 255.0
        vis = show_cam_on_image(image_array, sal_up)
        vis = np.uint8(255 * vis)
        vis_bgr = cv2.cvtColor(vis, cv2.COLOR_RGB2BGR)

        results.append({
            "heatmap_bgr": vis_bgr,
            "heatmap_rgb": cv2.cvtColor(vis_bgr, cv2.COLOR_BGR2RGB),
            "attribution": patch_saliency,
            "similarity":  baseline_sim,
        })

    return results


def generate_clip_margin_saliency(
    images: list,
    true_prompts: list,
    wrong_prompts: list,
    clip_ft,
    embed_modifier=None,
) -> list[dict]:
    """
    Occlusion saliency on the CLASS MARGIN (sim_true − sim_wrong), computed for
    the original AND the edited model in a single occlusion sweep.

    Why margin instead of a single prompt's similarity: for near-boundary images
    a feature-space edit shifts each class similarity by only ~0.1–1%, so
    single-prompt saliency maps look identical before/after editing. The
    decision, however, is made by the margin — which is exactly what the edit
    flips — so per-patch margin contributions expose the edit's effect directly.

    Both models are evaluated on the SAME masked forward pass (the modifier is
    applied to the same unnormalized embedding), which halves compute and
    guarantees the two maps differ only because of the edit.

    Parameters
    ----------
    images         : list of PIL Images
    true_prompts   : list[str] — prompt for the true class, one per image
    wrong_prompts  : list[str] — prompt for the wrong class, one per image
    clip_ft        : CLIPZeroShot — used for device / model-name lookup
    embed_modifier : callable applied after visual_projection (the ablation);
                     if None, attr_abl simply equals attr_orig.

    Returns
    -------
    List of dicts:
        "attr_orig" / "attr_abl" : signed (gs, gs) float arrays.
            attr[r,c] = baseline_margin − margin_with_patch_(r,c)_masked.
            Positive → masking the patch hurts the true class (patch supports
            the TRUE class); negative → patch supports the WRONG class.
        "margin_orig" / "margin_abl" : float baseline margins (no masking)
    """
    device = next(clip_ft.model.parameters()).device
    model_name = clip_ft.model_name

    if model_name not in _lrp_wrapper_cache:
        hf_name = _hf_model_path(_HF_MODEL_NAMES.get(model_name, "openai/clip-vit-base-patch32"))
        hf_model = CLIPModel.from_pretrained(hf_name, attn_implementation="eager").to(device)
        hf_processor = CLIPProcessor.from_pretrained(hf_name)
        _lrp_wrapper_cache[model_name] = CLIPLRPWrapper(hf_model, hf_processor)

    wrapper = _lrp_wrapper_cache[model_name]
    clip_model = wrapper.model
    clip_processor = wrapper.processor
    gs, ps = wrapper.grid_size, wrapper.patch_size

    results = []
    for image, p_true, p_wrong in zip(images, true_prompts, wrong_prompts):
        clip_image = image.convert("RGB")
        image_inputs = clip_processor(images=clip_image, return_tensors="pt")
        text_inputs  = clip_processor(text=[p_true, p_wrong], return_tensors="pt", padding=True)
        pixel_values = image_inputs["pixel_values"].to(device)

        with torch.no_grad():
            text_out = clip_model.text_model(
                input_ids=text_inputs["input_ids"].to(device),
                attention_mask=text_inputs["attention_mask"].to(device),
                return_dict=True,
            )
            txt = clip_model.text_projection(text_out.pooler_output)
            txt = txt / txt.norm(dim=-1, keepdim=True)            # (2, D): [true, wrong]

            def _margins(pv):
                vo  = clip_model.vision_model(pixel_values=pv, return_dict=True)
                emb = clip_model.visual_projection(vo.pooler_output)   # (1, D) unnormalized
                e_o = emb / emb.norm(dim=-1, keepdim=True)
                s_o = (e_o @ txt.T).squeeze(0)                          # (2,)
                m_o = (s_o[0] - s_o[1]).item()
                if embed_modifier is not None:
                    emb_a = embed_modifier(emb)
                    e_a   = emb_a / emb_a.norm(dim=-1, keepdim=True)
                    s_a   = (e_a @ txt.T).squeeze(0)
                    m_a   = (s_a[0] - s_a[1]).item()
                else:
                    m_a = m_o
                return m_o, m_a

            base_o, base_a = _margins(pixel_values)
            attr_o = np.zeros((gs, gs), dtype=np.float32)
            attr_a = np.zeros((gs, gs), dtype=np.float32)
            for r in range(gs):
                for c in range(gs):
                    pv_masked = pixel_values.clone()
                    pv_masked[:, :, r * ps:(r + 1) * ps, c * ps:(c + 1) * ps] = 0
                    m_o, m_a = _margins(pv_masked)
                    attr_o[r, c] = base_o - m_o
                    attr_a[r, c] = base_a - m_a

        results.append({
            "attr_orig":   attr_o,
            "attr_abl":    attr_a,
            "margin_orig": base_o,
            "margin_abl":  base_a,
        })

    return results


def generate_clip_heatmaps(
    images: list[Image.Image],
    prompts: list[str],
    clip_ft,
    method: str = "transformer_attribution",
    embed_modifier=None,
) -> list[dict]:
    """
    Generate LRP heatmaps for each (image, prompt) pair using the fine-tuned CLIP model.

    Parameters
    ----------
    images         : list of PIL Images
    prompts        : list of text strings, one per image
    clip_ft        : CLIPZeroShot — device and model name are derived from it
    method         : "transformer_attribution" (recommended) | "attention_rollout" | "gradient"
    embed_modifier : optional callable (unnorm_embed: Tensor) -> Tensor applied after
                     visual_projection but before L2-normalization. Use this to inject
                     feature-space ablations (e.g. SAE concept removal) into the LRP
                     attribution so that attention gradients reflect the ablated model.

    Returns
    -------
    List of dicts with keys:
        "heatmap_bgr"  : np.ndarray uint8 BGR overlay (for cv2.imwrite / imshow)
        "heatmap_rgb"  : np.ndarray uint8 RGB overlay (for plt.imshow)
        "attribution"  : np.ndarray float32 [grid_size, grid_size] raw scores
        "similarity"   : float cosine similarity score
    """
    device = next(clip_ft.model.parameters()).device
    model_name = clip_ft.model_name

    # Load or reuse the HuggingFace CLIP model required by CLIPLRPWrapper.
    # LRP attribution needs HuggingFace's eager attention implementation.
    if model_name not in _lrp_wrapper_cache:
        hf_name = _hf_model_path(_HF_MODEL_NAMES.get(model_name, "openai/clip-vit-base-patch32"))
        hf_model = CLIPModel.from_pretrained(hf_name, attn_implementation="eager").to(device)
        hf_processor = CLIPProcessor.from_pretrained(hf_name)
        _lrp_wrapper_cache[model_name] = CLIPLRPWrapper(hf_model, hf_processor)

    wrapper = _lrp_wrapper_cache[model_name]
    results = []

    for image, prompt in zip(images, prompts):
        image_rgb = image.convert("RGB")
        img_size = wrapper.image_size

        # Preprocess
        image_inputs = wrapper.processor(images=image_rgb, return_tensors="pt")
        text_inputs  = wrapper.processor(text=[prompt], return_tensors="pt", padding=True)

        pixel_values       = image_inputs["pixel_values"].to(device)
        text_input_ids     = text_inputs["input_ids"].to(device)
        text_attention_mask = text_inputs["attention_mask"].to(device)

        # Overlay base image
        image_np = np.array(image_rgb.resize((img_size, img_size))) / 255.0

        # Run attribution + visualization
        heatmap_bgr, similarity = wrapper.generate_lrp_visualization(
            pixel_values=pixel_values,
            text_input_ids=text_input_ids,
            text_attention_mask=text_attention_mask,
            original_image_np=image_np,
            method=method,
            embed_modifier=embed_modifier,
        )

        # Raw attribution map (no overlay) for downstream use
        attribution, _ = wrapper.generate_lrp_image_text_attribution(
            pixel_values, text_input_ids, text_attention_mask, method=method,
            embed_modifier=embed_modifier,
        )

        results.append({
            "heatmap_bgr": heatmap_bgr,
            "heatmap_rgb": cv2.cvtColor(heatmap_bgr, cv2.COLOR_BGR2RGB),
            "attribution": attribution.cpu().numpy(),
            "similarity": similarity,
        })

    return results

def find_spurious_concepts(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    activation_threshold=None,
    prevalence_threshold=0.30,
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
        # ensure keys always exist so plot functions don't KeyError on skipped groups
        group_top_sets[gid_mis]   = set()
        group_prevalence[gid_mis] = {}

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
        mean_rep = mis_reps.mean(axis=0)                                   # (L,) mean activation — for ranking

        eligible = np.where(prev > prevalence_threshold)[0]
        if len(eligible) == 0:
            report_lines.append(
                f"  No concepts exceed {prevalence_threshold*100:.0f}% prevalence — skipping."
            )
            continue

        # rank eligible concepts by mean activation (captures both magnitude and frequency)
        pool_size    = len(eligible) #int(len(eligible) * .3) #len(eligible) # if len(eligible) < 3 else int(len(eligible) * .3)
        # top_concepts = eligible[np.argsort(mean_prev[eligible])[::-1]][:top_n_concepts]  # for report
        mis_pool     = set(eligible[np.argsort(mean_rep[eligible])[::-1]][:pool_size].tolist())
        # mis_pool     = set(eligible[np.argsort(mean_prev[eligible])[::-1]][:pool_size].tolist())
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
        # low_aligned_mean = low_reps.mean(axis=0)
        # low_aligned_pool = set(np.argsort(low_aligned_mean)[:pool_size].tolist())

        ########Just new test: top misclassified concepts that are also low in the aligned low-bg group (intersection)########
        # mis_reps  = te_reps_np[misclassified]                               # (n_mis, L)
        n_aligned_low = low_reps.shape[0]
        low_prev      = (low_reps < activation_threshold).sum(axis=0) / n_aligned_low  # (L,) fraction — for filtering & diff
        low_eligible = np.where(low_prev > prevalence_threshold)[0]
        low_aligned_mean = low_reps.mean(axis=0)                                   # (L,) mean activation — for ranking
        low_aligned_pool = set(low_eligible[np.argsort(low_aligned_mean[low_eligible])[::-1]][:pool_size].tolist())

        # ── High-aligned pool: concepts appearing in the majority of high_reps ─
        n_aligned_high    = high_reps.shape[0]
        high_prev         = (high_reps > activation_threshold).sum(axis=0) / n_aligned_high
        high_eligible     = np.where(high_prev > prevalence_threshold)[0]
        high_aligned_pool = set(high_eligible[np.argsort(high_prev[high_eligible])[::-1]][:pool_size].tolist())

        candidate_concepts = list(mis_pool & low_aligned_pool & high_aligned_pool)
        if not candidate_concepts:
            candidate_concepts = list(mis_pool & low_aligned_pool)
            print("  No mis∩low∩high intersection — falling back to mis∩low_aligned.")
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
                sae_dir=sae_dir,
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
    
    return all_candidate_concepts, concept_source_group, concept_mis_prevalence


def find_spurious_concepts_binary(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    active_threshold=0.7,
    inactive_threshold=0.9,
    top_n_concepts=10,
):
    """
    Find spurious concepts using a direct binary active/inactive criterion:

      spurious = {c : active(c, mis) > active_threshold}
               ∩ {c : inactive(c, low_aligned) > inactive_threshold}

    where
      active(c, mis)          = fraction of misclassified images where latent c is non-zero
      inactive(c, low_aligned)= fraction of aligned-low-bg images where latent c is zero

    Because this uses sparse SAE latents (TopK-selected), non-zero means the SAE
    explicitly selected that concept for the image.

    Returns
    -------
    all_candidate_concepts  : list[int]
    concept_source_group    : dict[int, int]   cid → gid_mis
    concept_mis_prevalence  : dict[int, float] cid → active fraction in misclassified group
    """
    import shutil

    te_reps_np = te_results["sae_representations"].cpu().numpy()
    ft_reps_np = ft_results["sae_representations"].cpu().numpy()
    te_paths   = te_results["image_paths"]
    ft_paths   = ft_results["image_paths"]

    group_ids_arr = np.array([s[3] for s in test_ds.samples])

    MISALIGNED_GROUP_NAMES = {1: "landbird / water bg", 2: "waterbird / land bg"}
    GROUP_FOLDER            = {1: "land_bird_on_water",  2: "water_bird_on_land"}
    FT_CLASS_FOLDER         = {1: "water_birds",          2: "land_birds"}
    _aligned_high           = {1: 3, 2: 0}
    _aligned_low            = {1: 0, 2: 3}

    all_candidate_concepts = []
    concept_source_group   = {}
    concept_mis_prevalence = {}
    group_top_sets         = {}
    group_prevalence       = {}

    report_lines = [
        "Spurious Concept Analysis — binary active/inactive",
        f"active_threshold={active_threshold}  inactive_threshold={inactive_threshold}"
        f"  |  Top {top_n_concepts} concepts",
        "=" * 80,
    ]

    for gid_mis in [1, 2]:
        group_top_sets[gid_mis]   = set()
        group_prevalence[gid_mis] = {}

        g_indices = np.where(group_ids_arr == gid_mis)[0]

        group_ds = ManifestDataset.__new__(ManifestDataset)
        group_ds.samples         = [test_ds.samples[i] for i in g_indices]
        group_ds.clip_preprocess = clip_ft.preprocess
        group_ds.manifest_path   = test_ds.manifest_path

        print(f"\nClassifying Group {gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — {len(g_indices)} images …")
        zs_stats = clip_ft.run(dataset=group_ds, prompt_mode="shape", dataset_name="waterbirds")
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

        mis_reps  = te_reps_np[misclassified]
        low_reps  = te_reps_np[group_ids_arr == _aligned_low[gid_mis]]

        # fraction of misclassified images where concept is non-zero (active)
        active_rate = (mis_reps != 0).mean(axis=0)            # (L,)
        # fraction of aligned-low images where concept is zero (inactive)
        inactive_rate_low = (low_reps == 0).mean(axis=0)      # (L,)

        active_in_mis   = set(np.where(active_rate      > active_threshold)[0].tolist())
        inactive_in_low = set(np.where(inactive_rate_low > inactive_threshold)[0].tolist())
        spurious        = active_in_mis & inactive_in_low

        report_lines.append(
            f"  Active in >{active_threshold*100:.0f}% of misclassified: {len(active_in_mis)}\n"
            f"  Inactive in >{inactive_threshold*100:.0f}% of aligned-low: {len(inactive_in_low)}\n"
            f"  Intersection (spurious candidates): {len(spurious)}"
        )

        if not spurious:
            report_lines.append("  No concepts meet the active∩inactive criterion — skipping.")
            continue

        # rank by activity rate in misclassified (highest first)
        spurious_sorted = sorted(spurious, key=lambda c: active_rate[c], reverse=True)
        all_candidate_concepts.extend(spurious_sorted)
        for cid in spurious_sorted:
            concept_source_group[cid]   = gid_mis
            concept_mis_prevalence[cid] = float(active_rate[cid])

        top_concepts = spurious_sorted[:top_n_concepts]
        group_top_sets[gid_mis]   = set(top_concepts)
        group_prevalence[gid_mis] = {c: float(active_rate[c]) for c in top_concepts}

        report_lines += [
            f"\n{'Concept ID':<12}  {'Concept Name':<30}  "
            f"{'Active in Mis':>14}  {'Inactive in Low':>16}",
            "-" * 78,
        ]
        for cid in top_concepts:
            cname = vocab_names[concept_match_scores[:, cid].argmax()]
            report_lines.append(
                f"{cid:<12}  {cname:<30}  "
                f"{active_rate[cid]*100:>13.1f}%  {inactive_rate_low[cid]*100:>15.1f}%"
            )

        # ── Save example images per concept ──────────────────────────────────
        group_folder    = GROUP_FOLDER[gid_mis]
        ft_class_folder = FT_CLASS_FOLDER[gid_mis]

        for cid in top_concepts:
            cname    = vocab_names[concept_match_scores[:, cid].argmax()]
            test_out = os.path.join(sae_dir, "test_images", "spurious_binary", group_folder)
            ft_out   = os.path.join(sae_dir, "ft_images",  "spurious_binary", ft_class_folder)
            os.makedirs(test_out, exist_ok=True)
            os.makedirs(ft_out,   exist_ok=True)

            active_mis = sorted(
                [idx for idx in misclassified if te_reps_np[idx, cid] != 0],
                key=lambda idx: te_reps_np[idx, cid], reverse=True,
            )
            for idx in active_mis[:10]:
                shutil.copy2(te_paths[idx],
                             os.path.join(test_out, os.path.basename(te_paths[idx])))

            active_ft = sorted(
                [i for i in range(len(ft_paths)) if ft_reps_np[i, cid] != 0],
                key=lambda i: ft_reps_np[i, cid], reverse=True,
            )
            for i in active_ft[:10]:
                shutil.copy2(ft_paths[i],
                             os.path.join(ft_out, os.path.basename(ft_paths[i])))

            plot_path = plot_concept_top_images(
                cid=cid, cname=cname,
                group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]})",
                top10_te_indices=active_mis[:10],
                top10_ft_indices=active_ft[:10],
                te_paths=te_paths, ft_paths=ft_paths,
                te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
                test_save_dir=test_out, ft_save_dir=ft_out,
            )
            print(f"  Concept {cid} [{cname}]: test→{test_out}  ft→{ft_out}  plot→{plot_path}")

            # ── Lowest-activation (zero) images ──────────────────────────────
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

    # ── Build aligned-group sets for plot functions ───────────────────────────
    for gid_aln, gid_mis_pair in [(3, 1), (0, 2)]:
        paired = group_top_sets.get(gid_mis_pair, set())
        group_top_sets[gid_aln] = paired
        aln_reps = te_reps_np[group_ids_arr == gid_aln]
        group_prevalence[gid_aln] = (
            {c: float((aln_reps[:, c] != 0).mean()) for c in paired}
            if len(aln_reps) else {}
        )

    device         = next(clip_ft.model.parameters()).device
    group_ids_list = [s[3] for s in test_ds.samples]

    plot_misaligned_concept_images(
        results=te_results, group_ids=group_ids_list,
        group_top_sets=group_top_sets, group_prevalence=group_prevalence,
        concept_match_scores=concept_match_scores, vocab_names=vocab_names,
        sae_dir=sae_dir, clip_ft=clip_ft, device=device, top_n_concepts=top_n_concepts,
    )
    plot_misaligned_vs_fttrain_concept_images(
        te_results=te_results, ft_results=ft_results, group_ids_te=group_ids_list,
        group_top_sets=group_top_sets, group_prevalence=group_prevalence,
        concept_match_scores=concept_match_scores, vocab_names=vocab_names,
        sae_dir=sae_dir, clip_ft=clip_ft, device=device, top_n_concepts=top_n_concepts,
    )

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "spurious_concepts_binary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSpurious concept binary report saved to {report_path}")

    return all_candidate_concepts, concept_source_group, concept_mis_prevalence




def find_spurious_concepts_highmag(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    top_n_concepts=20,
):
    """High-Magnitude Concept Selection on misaligned+misclassified images.

    Steps
    -----
    1. τ = mean activation over *active* (non-zero) SAE latents across target images.
    2. P(c) = #{x : a_c(x) > τ}  — raw count of target images where concept c fires above τ.
    3. Return top-n_concepts by P(c) (highest frequency).

    Target images: groups 1 and 2 (misaligned) that are also misclassified by clip_ft.

    Returns
    -------
    all_candidate_concepts  : list[int]
    concept_source_group    : dict[int, int]    cid → source gid_mis
    concept_mis_prevalence  : dict[int, float]  cid → P(c) / n_target  (fraction)
    """
    te_reps_np    = te_results["sae_representations"].cpu().numpy()   # (N, L)
    group_ids_arr = np.array([s[3] for s in test_ds.samples])

    MISALIGNED_GROUP_NAMES = {1: "landbird / water bg", 2: "waterbird / land bg"}

    all_candidate_concepts = []
    concept_source_group   = {}
    concept_mis_prevalence = {}

    report_lines = [
        "High-Magnitude Concept Selection",
        f"Top {top_n_concepts} concepts per misaligned+misclassified group",
        "=" * 80,
    ]

    for gid_mis in [1, 2]:
        g_indices = np.where(group_ids_arr == gid_mis)[0]

        # ── Classify the misaligned group with clip_ft ────────────────────────
        group_ds = ManifestDataset.__new__(ManifestDataset)
        group_ds.samples         = [test_ds.samples[i] for i in g_indices]
        group_ds.clip_preprocess = clip_ft.preprocess
        group_ds.manifest_path   = test_ds.manifest_path

        print(f"\nClassifying Group {gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — {len(g_indices)} images …")
        stats     = clip_ft.run(dataset=group_ds, prompt_mode="shape", dataset_name="waterbirds")
        preds     = np.array(stats["predictions_shape"])
        labels    = np.array(stats["true_labels"])
        mis_local = np.where(preds != labels)[0]
        misclassified = g_indices[mis_local]
        n_mis = len(misclassified)
        print(f"  Misclassified: {n_mis} / {len(g_indices)}")

        report_lines.append(
            f"\n{'─'*80}\n"
            f"Group {gid_mis} — {MISALIGNED_GROUP_NAMES[gid_mis]}\n"
            f"Misclassified: {n_mis} / {len(g_indices)}"
        )

        if n_mis == 0:
            report_lines.append("  No misclassified images — skipping.")
            continue

        # ── Step 1: τ = mean of active (non-zero) activations ────────────────
        mis_reps  = te_reps_np[misclassified]              # (n_mis, L)
        active    = mis_reps[mis_reps > 0]
        tau       = float(active.mean()) if len(active) > 0 else 0.0
        print(f"  τ (mean active activation): {tau:.6f}")

        # ── Step 2: P(c) = #{x : a_c(x) > τ} ────────────────────────────────
        counts = (mis_reps > tau).sum(axis=0)              # (L,) int counts

        # ── Step 3: select top-n by count ────────────────────────────────────
        top_indices = np.argsort(counts)[::-1][:top_n_concepts]
        top_indices = [int(c) for c in top_indices if counts[c] > 0]

        report_lines += [
            f"  τ = {tau:.6f}",
            f"  {'Concept':<12}  {'Count':>7}  {'Fraction':>9}  {'Name'}",
            "  " + "-" * 60,
        ]
        for cid in top_indices:
            frac  = counts[cid] / n_mis
            cname = (
                vocab_names[concept_match_scores[:, cid].argmax()]
                if vocab_names is not None and concept_match_scores is not None
                else str(cid)
            )
            report_lines.append(f"  {cid:<12}  {counts[cid]:>7}  {frac*100:>8.1f}%  {cname}")
            if cid not in concept_source_group:
                concept_source_group[cid]   = gid_mis
                concept_mis_prevalence[cid] = float(frac)

        all_candidate_concepts.extend(c for c in top_indices if c not in all_candidate_concepts)

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "highmag_concepts_report.txt")
    os.makedirs(os.path.dirname(report_path), exist_ok=True)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"\nHigh-magnitude concept report saved to {report_path}")

    return all_candidate_concepts, concept_source_group, concept_mis_prevalence


def find_and_show_spurious_concepts_binary(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft,
    active_threshold=0.7,
    inactive_threshold=0.7,
    top_n_concepts=10,
    sae_model= None,
    device = None,
):
    """
    Find spurious concepts using a direct binary active/inactive criterion:

      spurious = {c : active(c, mis) > active_threshold}
               ∩ {c : inactive(c, low_aligned) > inactive_threshold}

    where
      active(c, mis)          = fraction of misclassified images where latent c is non-zero
      inactive(c, low_aligned)= fraction of aligned-low-bg images where latent c is zero

    Because this uses sparse SAE latents (TopK-selected), non-zero means the SAE
    explicitly selected that concept for the image.

    Returns
    -------
    all_candidate_concepts  : list[int]
    concept_source_group    : dict[int, int]   cid → gid_mis
    concept_mis_prevalence  : dict[int, float] cid → active fraction in misclassified group
    """
    import shutil

    te_reps_np = te_results["sae_representations"].cpu().numpy()
    ft_reps_np = ft_results["sae_representations"].cpu().numpy()
    te_paths   = te_results["image_paths"]
    ft_paths   = ft_results["image_paths"]

    if device is None:
        device = next(clip_ft.model.parameters()).device

    group_ids_arr = np.array([s[3] for s in test_ds.samples])

    MISALIGNED_GROUP_NAMES = {1: "landbird / water bg", 2: "waterbird / land bg"}
    GROUP_FOLDER            = {1: "land_bird_on_water",  2: "water_bird_on_land"}
    FT_CLASS_FOLDER         = {1: "water_birds",          2: "land_birds"}
    _aligned_high           = {1: 3, 2: 0}
    _aligned_low            = {1: 0, 2: 3}

    all_candidate_concepts = []
    concept_source_group   = {}
    concept_mis_prevalence = {}
    group_top_sets         = {}
    group_prevalence       = {}

    report_lines = [
        "Spurious Concept Analysis — binary active/inactive",
        f"active_threshold={active_threshold}  inactive_threshold={inactive_threshold}"
        f"  |  Top {top_n_concepts} concepts",
        "=" * 80,
    ]

    for gid_mis in [1, 2]:
        group_top_sets[gid_mis]   = set()
        group_prevalence[gid_mis] = {}

        g_indices = np.where(group_ids_arr == gid_mis)[0]

        group_ds = ManifestDataset.__new__(ManifestDataset)
        group_ds.samples         = [test_ds.samples[i] for i in g_indices]
        group_ds.clip_preprocess = clip_ft.preprocess
        group_ds.manifest_path   = test_ds.manifest_path

        print(f"\nClassifying Group {gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — {len(g_indices)} images …")
        zs_stats = clip_ft.run(dataset=group_ds, prompt_mode="shape", dataset_name="waterbirds")
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

        mis_reps  = te_reps_np[misclassified]
        low_reps  = te_reps_np[group_ids_arr == _aligned_low[gid_mis]]

        # fraction of misclassified images where concept is non-zero (active)
        active_rate = (mis_reps != 0).mean(axis=0)            # (L,)
        # fraction of aligned-low images where concept is zero (inactive)
        inactive_rate_low = (low_reps == 0).mean(axis=0)      # (L,)

        active_in_mis   = set(np.where(active_rate      > active_threshold)[0].tolist())
        inactive_in_low = set(np.where(inactive_rate_low > inactive_threshold)[0].tolist())
        spurious        = active_in_mis & inactive_in_low

        report_lines.append(
            f"  Active in >{active_threshold*100:.0f}% of misclassified: {len(active_in_mis)}\n"
            f"  Inactive in >{inactive_threshold*100:.0f}% of aligned-low: {len(inactive_in_low)}\n"
            f"  Intersection (spurious candidates): {len(spurious)}"
        )

        if not spurious:
            report_lines.append("  No concepts meet the active∩inactive criterion — skipping.")
            continue

        # rank by activity rate in misclassified (highest first)
        spurious_sorted = sorted(spurious, key=lambda c: active_rate[c], reverse=True)
        all_candidate_concepts.extend(spurious_sorted)
        for cid in spurious_sorted:
            concept_source_group[cid]   = gid_mis
            concept_mis_prevalence[cid] = float(active_rate[cid])

        top_concepts = spurious_sorted[:top_n_concepts]
        group_top_sets[gid_mis]   = set(top_concepts)
        group_prevalence[gid_mis] = {c: float(active_rate[c]) for c in top_concepts}

        report_lines += [
            f"\n{'Concept ID':<12}  {'Concept Name':<30}  "
            f"{'Active in Mis':>14}  {'Inactive in Low':>16}",
            "-" * 78,
        ]
        for cid in top_concepts:
            cname = vocab_names[concept_match_scores[:, cid].argmax()]
            report_lines.append(
                f"{cid:<12}  {cname:<30}  "
                f"{active_rate[cid]*100:>13.1f}%  {inactive_rate_low[cid]*100:>15.1f}%"
            )

        # ── Save example images per concept ──────────────────────────────────
        group_folder    = GROUP_FOLDER[gid_mis]
        ft_class_folder = FT_CLASS_FOLDER[gid_mis]

        

        for cid in top_concepts:
            cname    = vocab_names[concept_match_scores[:, cid].argmax()]
            test_out = os.path.join(sae_dir, "test_images", "spurious_binary", group_folder)
            ft_out   = os.path.join(sae_dir, "ft_images",  "spurious_binary", ft_class_folder)
            os.makedirs(test_out, exist_ok=True)
            os.makedirs(ft_out,   exist_ok=True)


            # visualize_concepts_on_images(
            #     image_paths=te_results["image_paths"],
            #     clip_ft=clip_ft,
            #     sae_model=sae_model,
            #     concept_ids=candidate_concepts,
            #     device=next(clip_ft.model.parameters()).device,
            #     concept_match_scores=concept_match_scores,
            #     vocab_names=vocab_names,
            # )
            active_mis = sorted(
                [idx for idx in misclassified if te_reps_np[idx, cid] != 0],
                key=lambda idx: te_reps_np[idx, cid], reverse=True,
            )
            for idx in active_mis[:10]:
                shutil.copy2(te_paths[idx],
                             os.path.join(test_out, os.path.basename(te_paths[idx])))
            top_mis_indices = active_mis[:10]
            toptest_paths   = [te_paths[idx] for idx in top_mis_indices]
            toptest_reps    = te_reps_np[top_mis_indices]
            visualize_concepts_on_images(
                image_paths=toptest_paths,
                clip_ft=clip_ft,
                sae_model=sae_model,
                concept_ids=cid,
                device=device,
                concept_match_scores=concept_match_scores,
                vocab_names=vocab_names,
                group_name=MISALIGNED_GROUP_NAMES[gid_mis],
                sae_representations=toptest_reps,
                save_dir=test_out,
            )

            active_ft = sorted(
                [i for i in range(len(ft_paths)) if ft_reps_np[i, cid] != 0],
                key=lambda i: ft_reps_np[i, cid], reverse=True,
            )
            for i in active_ft[:10]:
                shutil.copy2(ft_paths[i],
                             os.path.join(ft_out, os.path.basename(ft_paths[i])))

            top_ft_indices = active_ft[:10]
            topft_paths    = [ft_paths[i] for i in top_ft_indices]
            topft_reps     = ft_reps_np[top_ft_indices]
            visualize_concepts_on_images(
                image_paths=topft_paths,
                clip_ft=clip_ft,
                sae_model=sae_model,
                concept_ids=cid,
                device=device,
                concept_match_scores=concept_match_scores,
                vocab_names=vocab_names,
                group_name=f"FT — {MISALIGNED_GROUP_NAMES[gid_mis]}",
                sae_representations=topft_reps,
                save_dir=ft_out,
            )
            # ── Lowest-activation (zero) images ──────────────────────────────
            _te_grp_folders = {
                0: "landbird_land_bg", 1: "landbird_water_bg",
                2: "waterbird_land_bg", 3: "waterbird_water_bg",
            }
            _ft_lbl_folders = {0: "land_birds", 1: "water_birds"}

            # bottom_te = np.argsort(te_reps_np[:, cid])[:10]
            # bottom_ft = np.argsort(ft_reps_np[:, cid])[:10]

            # for idx in bottom_te:
            #     actual_gid    = test_ds.samples[idx][3]
            #     actual_folder = _te_grp_folders.get(actual_gid, f"group_{actual_gid}")
            #     dest = os.path.join(sae_dir, "test_images", "lowest", actual_folder)
            #     os.makedirs(dest, exist_ok=True)
            #     shutil.copy2(te_paths[idx],
            #                  os.path.join(dest, os.path.basename(te_paths[idx])))

            # for i in bottom_ft:
            #     actual_label  = ft_ds.samples[i][1]
            #     actual_folder = _ft_lbl_folders.get(actual_label, f"label_{actual_label}")
            #     dest = os.path.join(sae_dir, "ft_images", "lowest", actual_folder)
            #     os.makedirs(dest, exist_ok=True)
            #     shutil.copy2(ft_paths[i],
            #                  os.path.join(dest, os.path.basename(ft_paths[i])))

            # te_labels_low = [
            #     _te_grp_folders.get(test_ds.samples[idx][3], f"grp{test_ds.samples[idx][3]}")
            #     for idx in bottom_te
            # ]
            # ft_labels_low = [
            #     _ft_lbl_folders.get(ft_ds.samples[i][1], f"lbl{ft_ds.samples[i][1]}")
            #     for i in bottom_ft
            # ]

            # test_plot_dir = os.path.join(sae_dir, "test_images", "lowest", "plots")
            # ft_plot_dir   = os.path.join(sae_dir, "ft_images",   "lowest", "plots")
            # os.makedirs(test_plot_dir, exist_ok=True)
            # os.makedirs(ft_plot_dir,   exist_ok=True)
            # low_plot_path = plot_concept_top_images(
            #     cid=cid, cname=cname,
            #     group_label=f"Grp{gid_mis} ({MISALIGNED_GROUP_NAMES[gid_mis]}) — Lowest",
            #     top10_te_indices=bottom_te.tolist(),
            #     top10_ft_indices=bottom_ft.tolist(),
            #     te_paths=te_paths, ft_paths=ft_paths,
            #     te_reps_np=te_reps_np, ft_reps_np=ft_reps_np,
            #     test_save_dir=test_plot_dir, ft_save_dir=ft_plot_dir,
            #     te_row_title="Test (all) — lowest activation",
            #     ft_row_title="FT set — lowest activation",
            #     te_labels=te_labels_low,
            #     ft_labels=ft_labels_low,
            # )
            # print(f"  Concept {cid} [{cname}] lowest: "
            #       f"test→test_images/lowest/*  ft→ft_images/lowest/*  plot→{low_plot_path}")

    # ── Build aligned-group sets for plot functions ───────────────────────────
    for gid_aln, gid_mis_pair in [(3, 1), (0, 2)]:
        paired = group_top_sets.get(gid_mis_pair, set())
        group_top_sets[gid_aln] = paired
        aln_reps = te_reps_np[group_ids_arr == gid_aln]
        group_prevalence[gid_aln] = (
            {c: float((aln_reps[:, c] != 0).mean()) for c in paired}
            if len(aln_reps) else {}
        )

    device         = next(clip_ft.model.parameters()).device
    group_ids_list = [s[3] for s in test_ds.samples]

    # plot_misaligned_concept_images(
    #     results=te_results, group_ids=group_ids_list,
    #     group_top_sets=group_top_sets, group_prevalence=group_prevalence,
    #     concept_match_scores=concept_match_scores, vocab_names=vocab_names,
    #     sae_dir=sae_dir, clip_ft=clip_ft, device=device, top_n_concepts=top_n_concepts,
    # )
    # plot_misaligned_vs_fttrain_concept_images(
    #     te_results=te_results, ft_results=ft_results, group_ids_te=group_ids_list,
    #     group_top_sets=group_top_sets, group_prevalence=group_prevalence,
    #     concept_match_scores=concept_match_scores, vocab_names=vocab_names,
    #     sae_dir=sae_dir, clip_ft=clip_ft, device=device, top_n_concepts=top_n_concepts,
    # )

    report_text = "\n".join(report_lines) + "\n"
    report_path = os.path.join(sae_dir, "spurious_concepts_binary_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nSpurious concept binary report saved to {report_path}")

    
    return all_candidate_concepts, concept_source_group, concept_mis_prevalence







@torch.no_grad()
def extract_active_concepts(
    image_paths: list,
    clip_ft,
    sae_model,
    vocab_names=None,
    concept_match_scores=None,
):
    """
    For a list of image paths, extract each image's active SAE concepts.

    Parameters
    ----------
    image_paths          : list of file paths to images.
    clip_ft              : CLIPZeroShot fine-tuned model.
    sae_model            : trained SAE model.
    vocab_names          : optional list of concept name strings.
    concept_match_scores : optional (V, L) array — used to name each concept.

    Returns
    -------
    list of dicts, one per image:
        {
          "path"            : str,
          "sparse_latents"  : torch.Tensor (L,)  — full sparse latent vector,
          "active_indices"  : list[int]           — indices where latent > 0,
          "active_values"   : list[float]         — activation values at those indices,
          "active_names"    : list[str]            — concept names (if vocab provided),
        }
    """
    from PIL import Image as _PIL

    device = next(clip_ft.model.parameters()).device
    results = []

    for path in tqdm(image_paths, desc="  Extracting active concepts"):
        img  = _PIL.open(path).convert("RGB")
        inp  = clip_ft.preprocess(img).unsqueeze(0).to(device)

        features          = clip_ft.model.encode_image(inp).float()  # (1, D)
        sparse_latents, _ = sae_model.encode(features)               # (1, L)
        sparse_latents    = sparse_latents.squeeze(0).cpu()          # (L,)

        active_indices = sparse_latents.nonzero(as_tuple=True)[0].tolist()
        active_values  = sparse_latents[active_indices].tolist()

        if vocab_names is not None and concept_match_scores is not None:
            active_names = [
                vocab_names[concept_match_scores[:, cid].argmax()]
                for cid in active_indices
            ]
        else:
            active_names = [str(cid) for cid in active_indices]

        results.append({
            "path":           path,
            "sparse_latents": sparse_latents,
            "active_indices": active_indices,
            "active_values":  active_values,
            "active_names":   active_names,
        })

    return results


@torch.no_grad()
def candidate_selection(
    te_results,
    clip_ft,
    class_prompts: dict = None,
    k: int = 10,
    w: float = 0.3,
):
    """
    Algorithm 1 — Candidate Selection.

    Identifies images whose fine-tuned CLIP pseudo-label is likely wrong or
    uncertain, using two complementary criteria:
      - Centroid disagreement: the image is closer (cosine sim) to a different
        class's hybrid centroid than to its predicted class centroid.
      - k-NN disagreement: the majority label among the k nearest neighbors
        in embedding space differs from the pseudo-label.

    Parameters
    ----------
    te_results    : dict from extract_sae_representations — must contain
                    'clip_representations' (N, D) and 'image_paths'.
    clip_ft       : CLIPZeroShot fine-tuned model used for pseudo-labels and
                    text embeddings.
    class_prompts : dict {class_id: text_prompt}.  Defaults to waterbirds
                    landbird/waterbird prompts.
    k             : number of nearest neighbors.
    w             : weight on text embedding in the hybrid centroid (0 = pure
                    visual mean, 1 = pure text embedding).

    Returns
    -------
    M          : torch.BoolTensor (N,)  — True = candidate image
    Y_hat      : torch.LongTensor (N,)  — pseudo-labels from fine-tuned CLIP
    M_centroid : torch.BoolTensor (N,)  — centroid-disagreement mask
    M_knn      : torch.BoolTensor (N,)  — k-NN-disagreement mask
    """
    device = next(clip_ft.model.parameters()).device

    # ── E: L2-normalised image embeddings ────────────────────────────────────
    E = te_results["clip_representations"].to(device).float()
    E = F.normalize(E, dim=-1)   # (N, D)
    N, D = E.shape

    # ── T: L2-normalised text embeddings, one per class ───────────────────────
    if class_prompts is None:
        class_prompts = {0: "a photo of a landbird", 1: "a photo of a waterbird"}
    T = clip_ft.encode_text_prompts(class_prompts).float().to(device)  # (C, D)
    C = T.shape[0]

    # ── Ŷ: pseudo-labels from fine-tuned CLIP ─────────────────────────────────
    Y_hat = (E @ T.T).argmax(dim=-1)   # (N,)

    # ── Hybrid centroids: μc = (1−w)·visual_mean(c) + w·T[c] ─────────────────
    centroids = torch.zeros(C, D, device=device)
    for c in range(C):
        mask = Y_hat == c
        if mask.sum() == 0:
            centroids[c] = T[c]
        else:
            visual_mean = E[mask].mean(dim=0)          # (D,) — mean of unit vecs
            centroids[c] = (1 - w) * visual_mean + w * T[c]
    centroids = F.normalize(centroids, dim=-1)         # (C, D) unit vectors

    # ── M_centroid: nearest centroid disagƒrees with pseudo-label ─────────────
    nearest_centroid = (E @ centroids.T).argmax(dim=-1)   # (N,)
    M_centroid = nearest_centroid != Y_hat                 # (N,) bool

    # ── M_knn: k-NN majority vote disagrees with pseudo-label ────────────────
    sim_matrix = E @ E.T                                   # (N, N) : cosine similarity of every pair of images
    sim_matrix.fill_diagonal_(-float("inf"))               # exclude self
    knn_indices = sim_matrix.topk(k, dim=-1).indices       # (N, k)

    knn_labels  = Y_hat[knn_indices]                       # (N, k)
    vote_counts = torch.zeros(N, C, device=device)
    for c in range(C):
        vote_counts[:, c] = (knn_labels == c).float().sum(dim=-1)
    knn_pred = vote_counts.argmax(dim=-1)                  # (N,)
    M_knn = knn_pred != Y_hat                              # (N,) bool

    # ── M: union of both candidate sets ──────────────────────────────────────
    M = M_centroid | M_knn

    n_total     = N
    n_centroid  = int(M_centroid.sum())
    n_knn       = int(M_knn.sum())
    n_union     = int(M.sum())
    print(
        f"  Candidates: {n_union}/{n_total} total  "
        f"({n_centroid} centroid, {n_knn} knn, overlap={n_centroid + n_knn - n_union})"
    )

    return M, Y_hat, M_centroid, M_knn


@torch.no_grad()
def Generate_Concept_Pool(
    te_results, test_ds,
    clip_ft, sae_model,
    vocab_names=None,
    concept_match_scores=None,
    sae_dir=None,
    batch_size=64,
    n_workers=10,
):
    """
    Identify every SAE concept that causally affects at least one prediction,
    and record per-image which concepts flip that image's prediction.

    For each concept active in any test image, zero it out via
    make_sae_ablation_hook and re-classify ALL images where it is active.
    If the prediction flips for an image, the concept is added both to the
    global pool and to that image's per-image pool.

    Parallelism: images within each concept are processed in batches of
    `batch_size` (GPU) with `n_workers` threads loading images from disk.

    Filename convention:
        concept_pool_{clip_model}_{latent_dim}.npy          — global pool
        concept_pool_per_image_{clip_model}_{latent_dim}.json — per-image pool

    Returns
    -------
    concept_pool   : list[int]
        Sorted list of concept indices that flip at least one prediction.
    per_image_pool : dict {img_idx (int): list[int]}
        For each image index, the concepts that flip its prediction.
    """
    from concurrent.futures import ThreadPoolExecutor
    import json
    from PIL import Image as _PIL
    from routesae import RouteSAE
    is_routesae = isinstance(sae_model, RouteSAE)
    if is_routesae:
        from routesae_adapter import routesae_embeds_batched

    device     = next(clip_ft.model.parameters()).device
    clip_tag   = clip_ft.model_name.replace("/", "~")
    te_reps    = te_results["sae_representations"]  # (N, L)
    latent_dim = te_reps.shape[1]
    te_paths   = te_results["image_paths"]

    pool_fname    = f"concept_pool_{clip_tag}_{latent_dim}.npy"
    per_img_fname = f"concept_pool_per_image_{clip_tag}_{latent_dim}.json"
    pool_path     = os.path.join(sae_dir, pool_fname)    if sae_dir else None
    per_img_path  = os.path.join(sae_dir, per_img_fname) if sae_dir else None

    # ── Load from cache if both files exist ──────────────────────────────────
    if pool_path and os.path.isfile(pool_path) and per_img_path and os.path.isfile(per_img_path):
        concept_pool = np.load(pool_path).tolist()
        with open(per_img_path) as f:
            per_image_pool = {int(k): v for k, v in json.load(f).items()}
        print(f"  Loaded concept pool ({len(concept_pool)} concepts) from {pool_path}")
        print(f"  Loaded per-image pool ({len(per_image_pool)} images) from {per_img_path}")
        return concept_pool, per_image_pool

    # ── Encode class text prompts ─────────────────────────────────────────────
    CLASS_PROMPTS = {0: "a photo of a landbird", 1: "a photo of a waterbird"}
    txt = clip_ft.encode_text_prompts(CLASS_PROMPTS).float()

    def _load(path):
        return clip_ft.preprocess(_PIL.open(path).convert("RGB"))

    # ── Original predictions (batched + parallel image loading) ──────────────
    orig_preds = []
    with ThreadPoolExecutor(max_workers=n_workers) as io_pool:
        for b in tqdm(range(0, len(te_paths), batch_size), desc="  Original predictions"):
            batch_paths = te_paths[b:b + batch_size]
            imgs  = torch.stack(list(io_pool.map(_load, batch_paths))).to(device)
            feats = clip_ft.model.encode_image(imgs).float()
            feats = feats / feats.norm(dim=-1, keepdim=True)
            orig_preds.extend((feats @ txt.T).argmax(dim=-1).cpu().tolist())
    orig_preds = np.array(orig_preds)

    # ── Find all unique concepts active in at least one image ─────────────────
    active_mask    = te_reps > 0                                     # (N, L) bool
    candidate_cids = active_mask.any(dim=0).nonzero(as_tuple=True)[0].tolist()
    print(f"  {len(candidate_cids)} active concepts to test across {len(te_paths)} images")

    # ── Test each concept across ALL its active images (batched) ─────────────
    concept_pool_set = set()
    per_image_pool   = {}   # img_idx → list[cid]

    if is_routesae:
        # Flatten every (cid, img_idx) pair across ALL candidate concepts into
        # one list, then chunk THAT into batch_size-sized mega-batches -- each
        # is one forward pass through routesae_embeds_batched covering many
        # UNRELATED concept/image pairs at once (a different concept can be
        # zeroed per row via sample_concept_idx), instead of one pass per
        # concept. Each pass re-pays the router/layer-selection computation
        # regardless of how many pairs it covers, so batching across concepts
        # (not just within one concept's images) is what actually cuts total
        # forward passes -- from O(num_concepts) down to O(total_pairs / batch_size).
        all_pairs = [
            (cid, img_idx)
            for cid in candidate_cids
            for img_idx in active_mask[:, cid].nonzero(as_tuple=True)[0].tolist()
        ]
        print(f"  {len(all_pairs)} (concept, image) pairs to test "
              f"in batches of {batch_size}")

        with ThreadPoolExecutor(max_workers=n_workers) as io_pool:
            for b in tqdm(range(0, len(all_pairs), batch_size), desc="  Testing concepts (batched)"):
                chunk = all_pairs[b:b + batch_size]
                cids     = [c for c, _ in chunk]
                img_idxs = [i for _, i in chunk]
                imgs = torch.stack(
                    list(io_pool.map(_load, [te_paths[i] for i in img_idxs]))
                ).to(device)
                sample_concept_idx = torch.tensor(cids, dtype=torch.long, device=device)
                feats = routesae_embeds_batched(
                    sae_model, clip_ft.model, imgs, sample_concept_idx,
                ).float()
                abl_preds = (feats @ txt.T).argmax(dim=-1).cpu().tolist()
                for j, (cid, img_idx) in enumerate(chunk):
                    if abl_preds[j] != orig_preds[img_idx]:
                        concept_pool_set.add(cid)
                        per_image_pool.setdefault(img_idx, []).append(cid)

    else:
        with ThreadPoolExecutor(max_workers=n_workers) as io_pool:
            for cid in tqdm(candidate_cids, desc="  Testing concepts"):
                active_img_indices = active_mask[:, cid].nonzero(as_tuple=True)[0].tolist()
                if not active_img_indices:
                    continue

                hook_fn = make_sae_ablation_hook(sae_model, [cid], device)
                handle  = clip_ft.model.visual.register_forward_hook(hook_fn)
                try:
                    for b in range(0, len(active_img_indices), batch_size):
                        batch_idx = active_img_indices[b:b + batch_size]
                        imgs  = torch.stack(
                            list(io_pool.map(_load, [te_paths[i] for i in batch_idx]))
                        ).to(device)
                        feats = clip_ft.model.encode_image(imgs).float()
                        feats = feats / feats.norm(dim=-1, keepdim=True)
                        abl_preds = (feats @ txt.T).argmax(dim=-1).cpu().tolist()
                        for j, img_idx in enumerate(batch_idx):
                            if abl_preds[j] != orig_preds[img_idx]:
                                concept_pool_set.add(cid)
                                per_image_pool.setdefault(img_idx, []).append(cid)
                finally:
                    handle.remove()

    concept_pool = sorted(concept_pool_set)
    print(f"  Global pool: {len(concept_pool)} concepts")
    print(f"  Per-image pool: {len(per_image_pool)} images have ≥1 influential concept")

    # ── Save ──────────────────────────────────────────────────────────────────
    if sae_dir is not None:
        os.makedirs(sae_dir, exist_ok=True)

        np.save(pool_path, np.array(concept_pool, dtype=np.int64))
        print(f"  Global pool saved to {pool_path}")

        with open(per_img_path, "w") as f:
            json.dump({str(k): v for k, v in per_image_pool.items()}, f)
        print(f"  Per-image pool saved to {per_img_path}")

        lines = [f"Concept pool — {len(concept_pool)} causal concepts",
                 f"clip_model={clip_tag}  latent_dim={latent_dim}\n"]
        for cid in concept_pool:
            if vocab_names is not None and concept_match_scores is not None:
                name     = vocab_names[concept_match_scores[:, cid].argmax()]
                n_imgs   = sum(1 for v in per_image_pool.values() if cid in v)
                lines.append(f"  {cid:5d}  {name}  (flips {n_imgs} images)")
            else:
                lines.append(f"  {cid}")
        with open(pool_path.replace(".npy", ".txt"), "w") as f:
            f.write("\n".join(lines) + "\n")

    return concept_pool, per_image_pool



def make_projection_ablation_hook(dirs, lam=1.0):
    def hook(module, input, output):
        feat = output.float()
        for d in dirs:
            feat = feat - lam * (feat @ d).unsqueeze(-1) * d.unsqueeze(0)
        return feat.to(output.dtype)
    return hook


def make_sae_ablation_hook(sae_model, concept_idx, device, lambda_coef=1.0):
    """
    Returns a forward hook that routes CLIP's visual output through the SAE,
    scales the specified latent dimensions by (1 - lam), then decodes back to CLIP space.
    lam=1.0 fully zeros the concepts (original behaviour); lam=0.0 leaves them unchanged.

    Usage — build the hook once and pass it as ablation_hook:
        hook = make_sae_ablation_hook(sae_model, candidate_concepts, device, lam=0.5)
        ablate_spurious_concepts(..., ablation_hook=hook)
    """
    concept_idx = list(concept_idx)

    def hook(module, input, output):
        feat = output.float()                                        # (B, D) CLIP features
        latents, _ = sae_model.encode(feat)                         # (B, L) sparse latents
        latents = latents.clone()
        latents[:, concept_idx] = (1.0 - lambda_coef) * latents[:, concept_idx]
        reconstructed = sae_model.decode(latents)                   # (B, D) back to CLIP space
        return reconstructed.to(output.dtype)

    return hook


def projection_ablation_hook(P, lambda_coef=1.0):
    """
    Returns a forward hook that projects CLIP's visual output onto the
    orthogonal complement of P's subspace: feat - lambda_coef * (feat @ P).

    P is a precomputed (d, d) projection matrix -- from qr_decompose_concepts
    (pass Q @ Q.T) or pinv_projection_matrix (already returns P directly).
    Both describe the same kind of object (a symmetric idempotent projector
    onto the concept subspace), just built differently -- see
    --projection_method's help text for the tradeoff -- so this hook doesn't
    need to know which one produced P.

    Usage — build the hook once and pass it as ablation_hook:
        hook = projection_ablation_hook(P, lambda_coef=args.ablation_coefficient)
        ablate_spurious_concepts(..., ablation_hook=hook)
    """

    def hook(module, input, output):
        feat = output.float()                           # (B, D) CLIP features
        clean_feat = feat - lambda_coef * (feat @ P)
        return clean_feat.to(output.dtype)
    return hook


def plot_concept_heatmaps_for_corrected_image(
    image_path,
    ablated_concept_indices,
    sae_model,
    clip_ft,
    device,
    save_dir,
    vocab_names=None,
    concept_match_scores=None,
    top_n=5,
    precomputed_acts=None,
):
    """
    For a single corrected image, select up to top_n ablated concepts by their
    pre-ablation SAE activation, then delegate heatmap rendering to
    visualize_concepts_on_images (one PNG per concept saved to save_dir).

    Parameters
    ----------
    image_path             : str — path to the original (pre-ablation) image
    ablated_concept_indices: iterable of int — all concept indices that were ablated
    sae_model              : trained SAE
    clip_ft                : CLIPZeroShot with .preprocess and .model
    device                 : torch.device
    save_dir               : str — directory where per-concept PNGs are saved
    vocab_names            : optional list[str] for concept name lookup
    concept_match_scores   : optional np.ndarray (n_vocab, n_latents) for lookup
    top_n                  : max number of concepts to visualise (default 5)
    precomputed_acts       : optional (d_sae,) array/tensor — this image's row of
                             te_results["sae_representations"]. When given, skips
                             the CLIP+SAE forward pass and guarantees the same
                             activations used everywhere else in the pipeline.
    """
    from PIL import Image as _PIL

    # ── 1. Pre-ablation SAE activations ──────────────────────────────────────
    if precomputed_acts is not None:
        acts = (
            precomputed_acts.cpu().numpy()
            if torch.is_tensor(precomputed_acts) else np.asarray(precomputed_acts)
        )
    else:
        img_pil = _PIL.open(image_path).convert("RGB")
        with torch.no_grad():
            inp  = clip_ft.preprocess(img_pil).unsqueeze(0).to(device)
            feat = clip_ft.model.encode_image(inp).float()
            sparse_codes, _ = sae_model.encode(feat)   # (1, d_sae)
            acts = sparse_codes[0].cpu().numpy()        # (d_sae,)

    # ── 2. Select top_n ablated concepts by activation ───────────────────────
    concept_acts = [
        (int(cid), float(acts[cid]))
        for cid in ablated_concept_indices
        if acts[cid] > 0
    ]
    concept_acts.sort(key=lambda x: -x[1])
    selected = concept_acts[:top_n]

    if not selected:
        return False  # no active ablated concepts in this image

    top_cids = [cid for cid, _ in selected]
    # sae_representations shape (1, d_sae) so visualize_concepts_on_images can
    # annotate each panel with the actual activation value.
    sae_reps = acts[np.newaxis, :]                     # (1, d_sae)

    # ── 3. Delegate heatmap rendering to visualize_concepts_on_images ────────
    visualize_concepts_on_images(
        image_paths=[image_path],
        clip_ft=clip_ft,
        sae_model=sae_model,
        concept_ids=top_cids,
        device=device,
        concept_match_scores=concept_match_scores,
        vocab_names=vocab_names,
        max_images=1,
        save_dir=save_dir,
        sae_representations=sae_reps,
    )
    return True


def plot_ablated_concepts_single_image(
    image_path,
    ablated_concept_indices,
    sae_model,
    clip_ft,
    device,
    out_path,
    vocab_names=None,
    concept_match_scores=None,
    top_n=5,
    precomputed_acts=None,
):
    """
    For a single image, produce one consolidated PNG containing:
      - Column 0 : original image (no overlay)
      - Columns 1–N : heatmap overlay for each of the top_n ablated concepts,
                      ranked by their pre-ablation SAE activation in this image.

    Heatmaps reflect the ORIGINAL (pre-ablation) activations — they show where
    each spurious concept fired before it was deactivated.

    Uses concept_spatial_heatmap(..., return_only=True) for heatmap computation,
    matching the same gradient approach used elsewhere in the pipeline.

    Parameters
    ----------
    image_path             : str — path to the image
    ablated_concept_indices: iterable of int — all concept indices that were ablated
    sae_model              : trained SAE
    clip_ft                : CLIPZeroShot with .preprocess and .model
    device                 : torch.device
    out_path               : str — full path where the PNG is saved
    vocab_names            : optional list[str] for concept name lookup
    concept_match_scores   : optional np.ndarray (n_vocab, n_latents) for lookup
    top_n                  : max number of concept panels (default 5)
    precomputed_acts       : optional (d_sae,) array/tensor — this image's row of
                             te_results["sae_representations"]. When given, skips
                             the CLIP+SAE forward pass and guarantees the same
                             activations used everywhere else in the pipeline.
    """
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL
    from overcomplete.visualization.plot_utils import show, interpolate_cv2, get_image_dimensions

    img_pil = _PIL.open(image_path).convert("RGB")

    # ── 1. Pre-ablation SAE activations ──────────────────────────────────────
    if precomputed_acts is not None:
        acts = (
            precomputed_acts.cpu().numpy()
            if torch.is_tensor(precomputed_acts) else np.asarray(precomputed_acts)
        )
    else:
        with torch.no_grad():
            inp  = clip_ft.preprocess(img_pil).unsqueeze(0).to(device)
            feat = clip_ft.model.encode_image(inp).float()
            sparse_codes, _ = sae_model.encode(feat)   # (1, d_sae)
            acts = sparse_codes[0].cpu().numpy()        # (d_sae,)

    # ── 2. Top_n ablated concepts by activation ───────────────────────────────
    concept_acts = [
        (int(cid), float(acts[cid]))
        for cid in ablated_concept_indices
        if acts[cid] > 0
    ]
    concept_acts.sort(key=lambda x: -x[1])
    selected = concept_acts[:top_n]

    if not selected:
        return False

    # ── 3. Heatmap per concept via concept_spatial_heatmap ────────────────────
    panels = []   # list of (cid, act, cname, img_pil, heatmap_grid)
    for cid, act in selected:
        result = concept_spatial_heatmap(
            image_path=image_path,
            clip_ft=clip_ft,
            sae_model=sae_model,
            sae_representations=acts[np.newaxis, :],   # (1, d_sae) — this image's acts
            concept_id=cid,
            device=device,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            return_only=True,
        )
        if result is None:
            continue
        img_out, heatmap = result
        cname = (
            vocab_names[concept_match_scores[:, cid].argmax()]
            if vocab_names is not None and concept_match_scores is not None
            else str(cid)
        )
        panels.append((cid, act, cname, img_out, heatmap))

    if not panels:
        return False

    # ── 4. Single consolidated figure: original | heatmap × N ────────────────
    n_cols = 1 + len(panels)
    fig, axes = plt.subplots(1, n_cols, figsize=(3.0 * n_cols, 3.5), squeeze=False)

    # Column 0 — original image
    plt.sca(axes[0, 0])
    show(img_pil)
    axes[0, 0].axis("off")
    axes[0, 0].set_title("Original", fontsize=8)

    # Columns 1..N — heatmap overlays (pre-ablation activations)
    for col, (cid, act, cname, img_out, heatmap) in enumerate(panels):
        width, height = get_image_dimensions(img_out)
        heatmap_up    = interpolate_cv2(heatmap, (width, height))
        ax = axes[0, col + 1]
        plt.sca(ax)
        show(img_out)
        show(heatmap_up, cmap="jet", alpha=0.5)
        ax.axis("off")
        ax.set_title(f"{cid}:{cname}\nact={act:.2f}", fontsize=6)

    plt.suptitle("Top ablated concepts — pre-ablation activations", fontsize=8, y=1.01)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return True


def _hook_label(fn) -> str:
    """Return a human-readable name for an ablation hook callable.

    Handles both plain functions (make_projection_ablation_hook) and
    closures returned by a factory (make_sae_ablation_hook.<locals>.hook).
    """
    qn = getattr(fn, "__qualname__", None) or getattr(fn, "__name__", "unknown")
    # "make_sae_ablation_hook.<locals>.hook" → "make_sae_ablation_hook"
    return qn.split(".<locals>")[0]


@torch.no_grad()
def _write_ablation_report(
    group_ids, true_labels, orig_preds, ablated_preds,
    concept_idx, vocab_names, concept_match_scores,
    concept_source_group, concept_mis_prevalence,
    sae_dir, concept_extractor_name, hook_tag, report_suffix, clip_mode,
):
    """Shared report-formatting tail for both ablate_spurious_concepts (MSAE)
    and ablate_spurious_concepts_routesae — everything after ablated_preds is
    computed is architecture-agnostic (just group_ids/true_labels/preds
    arrays), so this is the single source of truth for the report format.

    Returns the path the report was written to.
    """
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
    per_group_orig    = {}
    per_group_ablated = {}
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        per_group_orig[gid]    = (orig_preds[mask]    == true_labels[mask]).mean()
        per_group_ablated[gid] = (ablated_preds[mask] == true_labels[mask]).mean()

    overall_orig    = (orig_preds    == true_labels).mean()
    overall_ablated = (ablated_preds == true_labels).mean()
    report_lines += [
        "-" * 74,
        f"{'Overall':<42}  {overall_orig*100:>8.1f}%  {overall_ablated*100:>10.1f}%  "
        f"{(overall_ablated - overall_orig)*100:>+6.1f}%",
    ]

    if clip_mode == "zeroshot":
        TRAIN_SIZES = {0: 3498, 1: 184, 2: 56, 3: 1057}
        _total = sum(TRAIN_SIZES.values())
        adj_orig    = sum(per_group_orig[g]    * TRAIN_SIZES[g] / _total for g in TRAIN_SIZES if g in per_group_orig)
        adj_ablated = sum(per_group_ablated[g] * TRAIN_SIZES[g] / _total for g in TRAIN_SIZES if g in per_group_ablated)
        wga_orig    = min(per_group_orig.values())
        wga_ablated = min(per_group_ablated.values())
        report_lines += [
            "-" * 74,
            f"{'Adj avg acc (train-weighted)':<42}  {adj_orig*100:>8.1f}%  {adj_ablated*100:>10.1f}%  "
            f"{(adj_ablated - adj_orig)*100:>+6.1f}%",
            f"{'Worst-group acc':<42}  {wga_orig*100:>8.1f}%  {wga_ablated*100:>10.1f}%  "
            f"{(wga_ablated - wga_orig)*100:>+6.1f}%",
        ]

    report_text = "\n".join(report_lines) + "\n"
    extractor_tag = concept_extractor_name or "unknown_extractor"
    report_fname  = f"ablation_report_{extractor_tag}__{hook_tag}{report_suffix}.txt"
    report_path   = os.path.join(sae_dir, report_fname)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"Ablation report saved to {report_path}")
    return report_path


@torch.no_grad()
def _write_combined_ablation_report(
    group_ids, true_labels, orig_preds, ablation_results,
    concept_idx, vocab_names, concept_match_scores,
    sae_dir, concept_extractor_name, report_suffix,
):
    """Write one report comparing both independent ablation methods."""
    if vocab_names is not None and concept_match_scores is not None:
        concept_names = [
            f"{cid}:{vocab_names[concept_match_scores[:, cid].argmax()]}"
            for cid in concept_idx
        ]
    else:
        concept_names = [str(cid) for cid in concept_idx]

    report_lines = [
        "Spurious concept ablation comparison",
        "=" * 100,
        f"Ablated concepts ({len(concept_idx)}): {', '.join(concept_names)}",
        "-" * 100,
        f"{'Group':<42}  {'Original':>9}  {'Deactivation':>13}  {'Δ Deact':>9}  "
        f"{'Projection':>11}  {'Δ Proj':>8}",
        "-" * 100,
    ]
    for gid in sorted(set(group_ids.tolist())):
        mask = group_ids == gid
        original_acc = (orig_preds[mask] == true_labels[mask]).mean()
        method_accs = {
            name: (preds[mask] == true_labels[mask]).mean()
            for name, preds in ablation_results.items()
        }
        report_lines.append(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid)):<35})  "
            f"{original_acc * 100:>8.1f}%  "
            f"{method_accs['deactivation'] * 100:>12.1f}%  "
            f"{(method_accs['deactivation'] - original_acc) * 100:>+8.1f}%  "
            f"{method_accs['projection'] * 100:>10.1f}%  "
            f"{(method_accs['projection'] - original_acc) * 100:>+7.1f}%"
        )

    original_acc = (orig_preds == true_labels).mean()
    overall_method_accs = {
        name: (preds == true_labels).mean()
        for name, preds in ablation_results.items()
    }
    report_lines += [
        "-" * 100,
        f"{'Overall':<42}  {original_acc * 100:>8.1f}%  "
        f"{overall_method_accs['deactivation'] * 100:>12.1f}%  "
        f"{(overall_method_accs['deactivation'] - original_acc) * 100:>+8.1f}%  "
        f"{overall_method_accs['projection'] * 100:>10.1f}%  "
        f"{(overall_method_accs['projection'] - original_acc) * 100:>+7.1f}%",
    ]

    report_text = "\n".join(report_lines) + "\n"
    report_fname = f"ablation_report_{concept_extractor_name}__both{report_suffix}.txt"
    report_path = os.path.join(sae_dir, report_fname)
    with open(report_path, "w") as f:
        f.write(report_text)
    print(report_text)
    print(f"Combined ablation report saved to {report_path}")
    return report_path


@torch.no_grad()
def ablate_spurious_concepts(
    clip_ft, sae_model, test_ds,
    spurious_concept_indices,
    device, sae_dir,
    ablation_hook=make_projection_ablation_hook,
    lambda_coefficient=1.0,
    vocab_names=None,
    concept_match_scores=None,
    concept_source_group=None,
    concept_mis_prevalence=None,
    concept_extractor_name=None,
    candidate_mask_only=False,
    candidate_mask=None,
    clip_mode=None,
    te_sae_reps=None,
    report_suffix="",
    write_report=True,
):
    """
    Zero out `spurious_concept_indices` SAE latents for every test image,
    reconstruct CLIP features from the ablated latents, re-classify, and
    compare per-group accuracy against the original fine-tuned CLIP predictions.

    te_sae_reps : optional Tensor (N_te, L) — te_results["sae_representations"],
        row-aligned with test_ds.samples. Passed to the per-image concept heatmap
        plots so they reuse the precomputed activations instead of recomputing.
    report_suffix : optional string appended to the report filename (before .txt),
        e.g. "__denoised" — so runs differing only in denoise_concepts don't
        overwrite each other's reports.

    Saves: sae_dir/ablation_report_{extractor}_{hook}.txt
            sae_dir/ablation_corrected_group{1,2}.png
    """
    import clip as openai_clip
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL

    # Flush any MPS backlog left over from upstream steps -- --clip_mode
    # zeroshot runs an extra full pass over test_ds before this function is
    # even called (the zero-shot accuracy report), which fine-tuned mode
    # skips. Without this, that pass's residual cached memory compounds with
    # this function's own loops and can push MPS over its limit sooner than
    # a fine-tuned run of the exact same size would.
    if torch.device(device).type == "mps":
        torch.mps.empty_cache()

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

    # MPS's caching allocator doesn't reliably release memory across many
    # small forward passes the way CUDA does -- left unchecked, a 5000+ image
    # per-image loop creeps up to "MPS backend out of memory" well before
    # actual peak usage justifies it. Periodic empty_cache() keeps it flat;
    # no-op cost on CUDA/CPU where this isn't an issue.
    is_mps = torch.device(device).type == "mps"

    
    print("MPS allocated:", torch.mps.current_allocated_memory() / 1e9, "GB")
    print("MPS driver allocated:", torch.mps.driver_allocated_memory() / 1e9, "GB")
    # ── Original predictions (no hook) ── store full logits for delta scoring
    # Batched (64/call, matching CLIPZeroShot.run's proven-working pattern) --
    # calling encode_image individually per image (5794 separate MPS forward
    # calls) measured a steady ~30MB/image growth in torch.mps.current_allocated_memory()
    # that neither empty_cache() nor synchronize()+empty_cache() touched at all,
    # eventually hitting the MPS ceiling. Batching cuts the call COUNT ~64x,
    # which is what actually correlated with the growth, not the data volume.
    BATCH_SIZE = 64
    all_paths = [s[0] for s in test_ds.samples]
    orig_logits = []
    for b in tqdm(range(0, len(all_paths), BATCH_SIZE), desc="  Original CLIP"):
        if is_mps and (b // BATCH_SIZE) % 10 == 0:
            print(f"  [{b}/{len(all_paths)}] MPS allocated: {torch.mps.current_allocated_memory() / 1e9:.3f} GB")
            print(f"  [{b}/{len(all_paths)}] MPS driver allocated: {torch.mps.driver_allocated_memory() / 1e9:.3f} GB")
        batch_paths = all_paths[b:b + BATCH_SIZE]
        imgs = torch.stack([
            clip_ft.preprocess(_PIL.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)
        feats = clip_ft.model.encode_image(imgs).float()
        feats = feats / feats.norm(dim=-1, keepdim=True)
        orig_logits.append((feats @ txt.T).cpu())
        if is_mps and (b // BATCH_SIZE) % 10 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()
    orig_logits = torch.cat(orig_logits, dim=0)     # (N, 2)
    orig_preds  = orig_logits.argmax(dim=1).numpy()

    # ── Ablated predictions ───────────────────────────────────────────────────
    # When candidate_mask_only=True, only images in candidate_mask get the hook;
    # the rest keep their original logits unchanged.
    sig  = inspect.signature(ablation_hook)
    hook = ablation_hook(dirs) if len(sig.parameters) == 1 else ablation_hook

    if candidate_mask_only and candidate_mask is not None:
        cand_indices = set(candidate_mask.nonzero(as_tuple=True)[0].tolist())
        ablated_logits = orig_logits.clone()                      # default: keep original
        handle = clip_ft.model.visual.register_forward_hook(hook)
        try:
            for i, (path, _, _, _) in enumerate(tqdm(test_ds.samples, desc="  Ablated CLIP (candidates only)")):
                if i not in cand_indices:
                    continue
                img  = _PIL.open(path).convert("RGB")
                inp  = clip_ft.preprocess(img).unsqueeze(0).to(device)
                feat = clip_ft.model.encode_image(inp).float()
                feat = feat / feat.norm(dim=-1, keepdim=True)
                ablated_logits[i] = (feat @ txt.T).squeeze(0).cpu()
                if is_mps and i % 50 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        finally:
            handle.remove()
    else:
        # Batched -- same reasoning as the "Original predictions" loop above.
        # The hook (make_sae_ablation_hook/projection_ablation_hook) operates
        # on the visual output tensor as a whole, so it's batch-size agnostic.
        ablated_logits = []
        handle = clip_ft.model.visual.register_forward_hook(hook)
        try:
            for b in tqdm(range(0, len(all_paths), BATCH_SIZE), desc="  Ablated CLIP"):
                batch_paths = all_paths[b:b + BATCH_SIZE]
                imgs = torch.stack([
                    clip_ft.preprocess(_PIL.open(p).convert("RGB")) for p in batch_paths
                ]).to(device)
                feats = clip_ft.model.encode_image(imgs).float()
                feats = feats / feats.norm(dim=-1, keepdim=True)
                ablated_logits.append((feats @ txt.T).cpu())
                if is_mps and (b // BATCH_SIZE) % 10 == 0:
                    torch.mps.synchronize()
                    torch.mps.empty_cache()
        finally:
            handle.remove()
        ablated_logits = torch.cat(ablated_logits, dim=0)          # (N, 2)

    ablated_preds = ablated_logits.argmax(dim=1).numpy()
    hook_tag = _hook_label(ablation_hook)

    report_writer = _write_ablation_report if write_report else None
    if report_writer is not None:
        report_writer(
        group_ids=group_ids, true_labels=true_labels,
        orig_preds=orig_preds, ablated_preds=ablated_preds,
        concept_idx=concept_idx, vocab_names=vocab_names,
        concept_match_scores=concept_match_scores,
        concept_source_group=concept_source_group,
        concept_mis_prevalence=concept_mis_prevalence,
        sae_dir=sae_dir, concept_extractor_name=concept_extractor_name,
        hook_tag=hook_tag, report_suffix=report_suffix,
        clip_mode=clip_mode,
        )

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
            f"Top {top_k} corrected images after ablation",
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
                f"true:{true_cls}\norig:{orig_cls}",
                fontsize=7,
            )
        plt.tight_layout()
        fpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_corrected_group{gid}.png")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved corrected plot (group {gid}): {fpath}")

        # LRP heatmaps for corrected images — prompt = true class name
        corrected_images = [_PIL.open(test_ds.samples[i][0]).convert("RGB") for i in top_idx]
        corrected_prompts = [CLASS_NAMES[true_labels[i]] for i in top_idx]
        with torch.enable_grad():
            heatmap_results = generate_clip_heatmaps(corrected_images, corrected_prompts, clip_ft)
        # _ablation_embed_modifier wraps the ablation hook as a plain callable applied
        # after visual_projection, before L2-normalization (inside the HF pipeline).
        def _ablation_embed_modifier(unnorm_embed):
            # hook signature is (module, input, output); simulate by passing None for module/input
            result = hook(None, None, unnorm_embed)
            return result if result is not None else unnorm_embed

        # ── DISABLED: LRP heatmap of the ablated model ────────────────────────
        # The edit is applied to the FINAL embedding, after all attention layers.
        # LRP/transformer-attribution reads the attention weights, which are
        # identical before and after the edit, so the ablated LRP map is
        # ~indistinguishable from the original (measured corr 0.9995 on a real
        # run). Kept for reference — to restore, uncomment this block and the
        # 3-row figure below.
        # with torch.enable_grad():
        #     ablated_heatmap_results = generate_clip_heatmaps(
        #         corrected_images, corrected_prompts, clip_ft,
        #         embed_modifier=_ablation_embed_modifier,
        #     )
        # fig, axes = plt.subplots(3, top_k, figsize=(2.5 * top_k, 7.5), squeeze=False)
        # fig.suptitle(
        #     f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — corrected: original | orig heatmap | ablated heatmap",
        #     fontsize=10, fontweight="bold",
        # )
        # for col, (img_pil, hr, ahr) in enumerate(zip(corrected_images, heatmap_results, ablated_heatmap_results)):
        #     axes[0, col].imshow(img_pil)
        #     axes[0, col].axis("off")
        #     axes[0, col].set_title(corrected_prompts[col], fontsize=7)
        #     axes[1, col].imshow(hr["heatmap_rgb"])
        #     axes[1, col].axis("off")
        #     axes[1, col].set_title(f"orig sim={hr['similarity']:.2f}", fontsize=7)
        #     axes[2, col].imshow(ahr["heatmap_rgb"])
        #     axes[2, col].axis("off")
        #     axes[2, col].set_title(f"abl sim={ahr['similarity']:.2f}", fontsize=7)

        # Occlusion saliency: masks one patch at a time and re-runs the FULL
        # (edited) pipeline per patch, so original vs ablated maps genuinely
        # reflect the behavioral difference introduced by the edit.
        saliency_orig = generate_clip_saliency(corrected_images, corrected_prompts, clip_ft)
        saliency_abl  = generate_clip_saliency(
            corrected_images, corrected_prompts, clip_ft,
            embed_modifier=_ablation_embed_modifier,
        )
        fig, axes = plt.subplots(4, top_k, figsize=(2.5 * top_k, 10), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — corrected: "
            f"original | LRP (orig) | occlusion (orig) | occlusion (ablated)",
            fontsize=10, fontweight="bold",
        )
        # logit_scale (~100) makes softmax over cosine sims meaningful — raw
        # sims are so close together a plain softmax would always read ~0.50.
        logit_scale = clip_ft.model.logit_scale.exp().item()
        for col, (img_pil, hr, so, sa) in enumerate(
            zip(corrected_images, heatmap_results, saliency_orig, saliency_abl)
        ):
            idx = top_idx[col]
            t   = true_labels[idx]
            ol, al = orig_logits_np[idx], ablated_logits_np[idx]   # (2,) each: [land, water]
            po = np.exp(logit_scale * (ol - ol.max())); po /= po.sum()
            pa = np.exp(logit_scale * (al - al.max())); pa /= pa.sum()
            axes[0, col].imshow(img_pil)
            axes[0, col].axis("off")
            axes[0, col].set_title(corrected_prompts[col], fontsize=7)
            axes[1, col].imshow(hr["heatmap_rgb"])
            axes[1, col].axis("off")
            axes[1, col].set_title(f"LRP sim={hr['similarity']:.2f}", fontsize=7)
            axes[2, col].imshow(so["heatmap_rgb"])
            axes[2, col].axis("off")
            axes[2, col].set_title(
                f"L={ol[0]:.3f} W={ol[1]:.3f}\np(true)={po[t]:.2f}", fontsize=6
            )
            axes[3, col].imshow(sa["heatmap_rgb"])
            axes[3, col].axis("off")
            axes[3, col].set_title(
                f"L={al[0]:.3f} W={al[1]:.3f}\np(true)={pa[t]:.2f}", fontsize=6
            )
        plt.tight_layout()
        hfpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_corrected_heatmap_group{gid}.png")
        os.makedirs(os.path.dirname(hfpath), exist_ok=True)
        plt.savefig(hfpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved corrected heatmap (group {gid}): {hfpath}")

        # ── Margin-based occlusion: evidence maps split by decision direction ─
        # The signed margin attribution is split into its positive part
        # (patches whose masking HURTS the true class → evidence FOR the true
        # class) and its negative part (patches whose masking HELPS the true
        # class → evidence FOR the wrong class, i.e. the spurious regions).
        # Rows: image | FOR-true (orig) | FOR-wrong (orig) | FOR-wrong (ablated).
        # Rows 2 and 3 share a color scale so the ablation's reduction of
        # wrong-class evidence is directly comparable.
        import torch.nn.functional as _F
        true_pr  = [CLASS_PROMPTS[true_labels[i]] for i in top_idx]
        wrong_pr = [CLASS_PROMPTS[1 - true_labels[i]] for i in top_idx]
        margin_res = generate_clip_margin_saliency(
            corrected_images, true_pr, wrong_pr, clip_ft,
            embed_modifier=_ablation_embed_modifier,
        )
        fig, axes = plt.subplots(4, top_k, figsize=(2.5 * top_k, 10), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — corrected, margin occlusion:\n"
            f"image | evidence FOR true class (orig) | evidence FOR wrong class (orig) "
            f"| edit acts here (removed component)",
            fontsize=10, fontweight="bold",
        )
        for col, (img_pil, mr) in enumerate(zip(corrected_images, margin_res)):
            idx = top_idx[col]
            t   = true_labels[idx]
            ol, al = orig_logits_np[idx], ablated_logits_np[idx]
            po = np.exp(logit_scale * (ol - ol.max())); po /= po.sum()
            pa = np.exp(logit_scale * (al - al.max())); pa /= pa.sum()
            ao, aa = mr["attr_orig"], mr["attr_abl"]
            pos_o = np.clip(ao, 0, None)      # orig: pushes toward true class
            neg_o = np.clip(-ao, 0, None)     # orig: pushes toward wrong class
            # per-patch edit contribution: hiding these patches weakens the
            # edit's effect on the margin → they carry the removed component.
            # (Showing the raw ablated map instead is uninformative: it is
            # ~identical to the original because the edit shifts the margin
            # near-uniformly across maskings.)
            edit_map = np.clip(aa - ao, 0, None)
            vmax_pos  = float(pos_o.max()) or 1.0
            vmax_neg  = float(neg_o.max()) or 1.0
            vmax_edit = float(edit_map.max()) or 1.0   # own scale — tiny magnitudes
            im224 = img_pil.resize((224, 224))
            _up = lambda a: _F.interpolate(
                torch.tensor(a[None, None]), size=(224, 224),
                mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            axes[0, col].imshow(im224)
            axes[0, col].axis("off")
            axes[0, col].set_title(corrected_prompts[col], fontsize=7)
            axes[1, col].imshow(im224)
            axes[1, col].imshow(_up(pos_o), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_pos)
            axes[1, col].axis("off")
            axes[1, col].set_title(f"p(true)={po[t]:.2f}", fontsize=7)
            axes[2, col].imshow(im224)
            axes[2, col].imshow(_up(neg_o), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_neg)
            axes[2, col].axis("off")
            axes[2, col].set_title(f"p(true)={po[t]:.2f}", fontsize=7)
            axes[3, col].imshow(im224)
            axes[3, col].imshow(_up(edit_map), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_edit)
            axes[3, col].axis("off")
            axes[3, col].set_title(f"p(true)={pa[t]:.2f}", fontsize=7)
        plt.tight_layout()
        mfpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_corrected_margin_heatmap_group{gid}.png")
        plt.savefig(mfpath, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  Saved margin occlusion heatmap (group {gid}): {mfpath}")

        # ── Per-image concept ablation heatmaps ──────────────────────────────
        # plot_concept_heatmaps_for_corrected_image: one PNG per concept (per-concept view)
        # plot_ablated_concepts_single_image:        one consolidated PNG per image
        base_concept_dir = os.path.join(
            sae_dir, concept_extractor_name, hook_tag,
            f"concept_heatmaps_group{gid}",
        )
        n_with_heatmaps = 0
        for rank, img_idx in enumerate(top_idx):
            img_path = test_ds.samples[img_idx][0]
            img_acts = te_sae_reps[img_idx] if te_sae_reps is not None else None

            # per-concept PNGs (one file per concept)
            plot_concept_heatmaps_for_corrected_image(
                image_path=img_path,
                ablated_concept_indices=concept_idx,
                sae_model=sae_model,
                clip_ft=clip_ft,
                device=device,
                save_dir=os.path.join(base_concept_dir, f"img{img_idx:05d}_rank{rank}"),
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                top_n=5,
                precomputed_acts=img_acts,
            )

            # consolidated single PNG: original + 5 concept heatmaps side by side
            saved = plot_ablated_concepts_single_image(
                image_path=img_path,
                ablated_concept_indices=concept_idx,
                sae_model=sae_model,
                clip_ft=clip_ft,
                device=device,
                out_path=os.path.join(
                    base_concept_dir, f"img{img_idx:05d}_rank{rank}_consolidated.png"
                ),
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                top_n=5,
                precomputed_acts=img_acts,
            )
            if saved:
                n_with_heatmaps += 1
        if n_with_heatmaps > 0:
            print(f"  Saved per-image concept heatmaps (group {gid}): "
                  f"{n_with_heatmaps}/{len(top_idx)} images → {base_concept_dir}/")
        else:
            print(f"  Group {gid}: no ablated concept is active in any of the "
                  f"{len(top_idx)} top corrected images — no concept heatmaps saved.")

        # Occlusion saliency for corrected images — compare with LRP above
        # saliency_results = generate_clip_saliency(corrected_images, corrected_prompts, clip_ft)
        # fig, axes = plt.subplots(2, top_k, figsize=(2.5 * top_k, 5), squeeze=False)
        # fig.suptitle(
        #     f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — corrected: original | saliency",
        #     fontsize=10, fontweight="bold",
        # )
        # for col, (img_pil, sr) in enumerate(zip(corrected_images, saliency_results)):
        #     axes[0, col].imshow(img_pil)
        #     axes[0, col].axis("off")
        #     axes[0, col].set_title(corrected_prompts[col], fontsize=7)
        #     axes[1, col].imshow(sr["heatmap_rgb"])
        #     axes[1, col].axis("off")
        #     axes[1, col].set_title(f"sim={sr['similarity']:.2f}", fontsize=7)
        # plt.tight_layout()
        # sfpath = os.path.join(sae_dir, concept_extractor_name, ablation_hook.__qualname__, f"ablation_corrected_saliency_group{gid}.png")
        # os.makedirs(os.path.dirname(sfpath), exist_ok=True)
        # plt.savefig(sfpath, bbox_inches="tight", dpi=120)
        # plt.close()
        # print(f"  Saved corrected saliency (group {gid}): {sfpath}")

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
        fpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_harmed_group{gid}.png")
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        plt.savefig(fpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved harmed plot (group {gid}): {fpath}")

        # LRP heatmaps for harmed images — prompt = true class name
        harmed_images  = [_PIL.open(test_ds.samples[i][0]).convert("RGB") for i in top_idx]
        harmed_prompts = [CLASS_NAMES[true_labels[i]] for i in top_idx]
        with torch.enable_grad():
            heatmap_results = generate_clip_heatmaps(harmed_images, harmed_prompts, clip_ft)
        # _ablation_embed_modifier wraps the ablation hook as a plain callable applied
        # after visual_projection, before L2-normalization (inside the HF pipeline).
        def _ablation_embed_modifier(unnorm_embed):
            result = hook(None, None, unnorm_embed)
            return result if result is not None else unnorm_embed

        # ── DISABLED: LRP heatmap of the ablated model ────────────────────────
        # See the corrected-images section above for the full explanation: the
        # edit happens after the attention layers, so the ablated LRP map is
        # ~identical to the original (corr 0.9995 measured). Kept for reference.
        # with torch.enable_grad():
        #     ablated_heatmap_results = generate_clip_heatmaps(
        #         harmed_images, harmed_prompts, clip_ft,
        #         embed_modifier=_ablation_embed_modifier,
        #     )
        # fig, axes = plt.subplots(3, top_k, figsize=(2.5 * top_k, 7.5), squeeze=False)
        # fig.suptitle(
        #     f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — harmed: original | orig heatmap | ablated heatmap",
        #     fontsize=10, fontweight="bold",
        # )
        # for col, (img_pil, hr, ahr) in enumerate(zip(harmed_images, heatmap_results, ablated_heatmap_results)):
        #     axes[0, col].imshow(img_pil)
        #     axes[0, col].axis("off")
        #     axes[0, col].set_title(harmed_prompts[col], fontsize=7)
        #     axes[1, col].imshow(hr["heatmap_rgb"])
        #     axes[1, col].axis("off")
        #     axes[1, col].set_title(f"orig sim={hr['similarity']:.2f}", fontsize=7)
        #     axes[2, col].imshow(ahr["heatmap_rgb"])
        #     axes[2, col].axis("off")
        #     axes[2, col].set_title(f"abl sim={ahr['similarity']:.2f}", fontsize=7)

        # Occlusion saliency: masks one patch at a time and re-runs the FULL
        # (edited) pipeline per patch, so original vs ablated maps genuinely
        # reflect the behavioral difference introduced by the edit.
        saliency_orig = generate_clip_saliency(harmed_images, harmed_prompts, clip_ft)
        saliency_abl  = generate_clip_saliency(
            harmed_images, harmed_prompts, clip_ft,
            embed_modifier=_ablation_embed_modifier,
        )
        fig, axes = plt.subplots(4, top_k, figsize=(2.5 * top_k, 10), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — harmed: "
            f"original | LRP (orig) | occlusion (orig) | occlusion (ablated)",
            fontsize=10, fontweight="bold",
        )
        # logit_scale (~100) makes softmax over cosine sims meaningful — raw
        # sims are so close together a plain softmax would always read ~0.50.
        logit_scale = clip_ft.model.logit_scale.exp().item()
        for col, (img_pil, hr, so, sa) in enumerate(
            zip(harmed_images, heatmap_results, saliency_orig, saliency_abl)
        ):
            idx = top_idx[col]
            t   = true_labels[idx]
            ol, al = orig_logits_np[idx], ablated_logits_np[idx]   # (2,) each: [land, water]
            po = np.exp(logit_scale * (ol - ol.max())); po /= po.sum()
            pa = np.exp(logit_scale * (al - al.max())); pa /= pa.sum()
            axes[0, col].imshow(img_pil)
            axes[0, col].axis("off")
            axes[0, col].set_title(harmed_prompts[col], fontsize=7)
            axes[1, col].imshow(hr["heatmap_rgb"])
            axes[1, col].axis("off")
            axes[1, col].set_title(f"LRP sim={hr['similarity']:.2f}", fontsize=7)
            axes[2, col].imshow(so["heatmap_rgb"])
            axes[2, col].axis("off")
            axes[2, col].set_title(
                f"L={ol[0]:.3f} W={ol[1]:.3f}\np(true)={po[t]:.2f}", fontsize=6
            )
            axes[3, col].imshow(sa["heatmap_rgb"])
            axes[3, col].axis("off")
            axes[3, col].set_title(
                f"L={al[0]:.3f} W={al[1]:.3f}\np(true)={pa[t]:.2f}", fontsize=6
            )
        plt.tight_layout()
        hfpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_harmed_heatmap_group{gid}.png")
        os.makedirs(os.path.dirname(hfpath), exist_ok=True)
        plt.savefig(hfpath, bbox_inches="tight", dpi=120)
        plt.close()
        print(f"  Saved harmed heatmap (group {gid}): {hfpath}")

        # ── Margin-based occlusion: evidence maps split by decision direction ─
        # See the corrected-images section for the full rationale.
        # Rows: image | FOR-true (orig) | FOR-wrong (orig) | FOR-wrong (ablated).
        import torch.nn.functional as _F
        true_pr  = [CLASS_PROMPTS[true_labels[i]] for i in top_idx]
        wrong_pr = [CLASS_PROMPTS[1 - true_labels[i]] for i in top_idx]
        margin_res = generate_clip_margin_saliency(
            harmed_images, true_pr, wrong_pr, clip_ft,
            embed_modifier=_ablation_embed_modifier,
        )
        fig, axes = plt.subplots(4, top_k, figsize=(2.5 * top_k, 10), squeeze=False)
        fig.suptitle(
            f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — harmed, margin occlusion:\n"
            f"image | evidence FOR true class (orig) | evidence FOR wrong class (orig) "
            f"| edit acts here (removed component)",
            fontsize=10, fontweight="bold",
        )
        for col, (img_pil, mr) in enumerate(zip(harmed_images, margin_res)):
            idx = top_idx[col]
            t   = true_labels[idx]
            ol, al = orig_logits_np[idx], ablated_logits_np[idx]
            po = np.exp(logit_scale * (ol - ol.max())); po /= po.sum()
            pa = np.exp(logit_scale * (al - al.max())); pa /= pa.sum()
            ao, aa = mr["attr_orig"], mr["attr_abl"]
            pos_o = np.clip(ao, 0, None)      # orig: pushes toward true class
            neg_o = np.clip(-ao, 0, None)     # orig: pushes toward wrong class
            # per-patch edit contribution — see corrected section for rationale.
            # Sign flipped vs the corrected section: for harmed images the edit
            # LOWERS the margin, so the patches responsible for the damage are
            # where hiding them makes the edit's (negative) effect weaker,
            # i.e. attr_orig − attr_abl > 0. Shows where the edit removed
            # evidence the model NEEDED (the collateral damage).
            edit_map = np.clip(ao - aa, 0, None)
            vmax_pos  = float(pos_o.max()) or 1.0
            vmax_neg  = float(neg_o.max()) or 1.0
            vmax_edit = float(edit_map.max()) or 1.0   # own scale — tiny magnitudes
            im224 = img_pil.resize((224, 224))
            _up = lambda a: _F.interpolate(
                torch.tensor(a[None, None]), size=(224, 224),
                mode="bilinear", align_corners=False,
            )[0, 0].numpy()
            axes[0, col].imshow(im224)
            axes[0, col].axis("off")
            axes[0, col].set_title(harmed_prompts[col], fontsize=7)
            axes[1, col].imshow(im224)
            axes[1, col].imshow(_up(pos_o), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_pos)
            axes[1, col].axis("off")
            axes[1, col].set_title(f"p(true)={po[t]:.2f}", fontsize=7)
            axes[2, col].imshow(im224)
            axes[2, col].imshow(_up(neg_o), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_neg)
            axes[2, col].axis("off")
            axes[2, col].set_title(f"p(true)={po[t]:.2f}", fontsize=7)
            axes[3, col].imshow(im224)
            axes[3, col].imshow(_up(edit_map), cmap="jet", alpha=0.5, vmin=0, vmax=vmax_edit)
            axes[3, col].axis("off")
            axes[3, col].set_title(f"p(true)={pa[t]:.2f}", fontsize=7)
        plt.tight_layout()
        mfpath = os.path.join(sae_dir, concept_extractor_name, hook_tag, f"ablation_harmed_margin_heatmap_group{gid}.png")
        plt.savefig(mfpath, bbox_inches="tight", dpi=120)
        plt.close(fig)
        print(f"  Saved margin occlusion heatmap (harmed group {gid}): {mfpath}")

        # ── Per-image concept ablation heatmaps for harmed images ─────────────
        base_concept_dir = os.path.join(
            sae_dir, concept_extractor_name, hook_tag,
            f"concept_heatmaps_harmed_group{gid}",
        )
        n_with_heatmaps = 0
        for rank, img_idx in enumerate(top_idx):
            img_path = test_ds.samples[img_idx][0]
            img_acts = te_sae_reps[img_idx] if te_sae_reps is not None else None

            plot_concept_heatmaps_for_corrected_image(
                image_path=img_path,
                ablated_concept_indices=concept_idx,
                sae_model=sae_model,
                clip_ft=clip_ft,
                device=device,
                save_dir=os.path.join(base_concept_dir, f"img{img_idx:05d}_rank{rank}"),
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                top_n=5,
                precomputed_acts=img_acts,
            )

            saved = plot_ablated_concepts_single_image(
                image_path=img_path,
                ablated_concept_indices=concept_idx,
                sae_model=sae_model,
                clip_ft=clip_ft,
                device=device,
                out_path=os.path.join(
                    base_concept_dir, f"img{img_idx:05d}_rank{rank}_consolidated.png"
                ),
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                top_n=5,
                precomputed_acts=img_acts,
            )
            if saved:
                n_with_heatmaps += 1
        if n_with_heatmaps > 0:
            print(f"  Saved per-image concept heatmaps (harmed group {gid}): "
                  f"{n_with_heatmaps}/{len(top_idx)} images → {base_concept_dir}/")
        else:
            print(f"  Group {gid}: no ablated concept is active in any of the "
                  f"{len(top_idx)} top harmed images — no concept heatmaps saved.")

    return orig_preds, ablated_preds, hook_tag

        # Occlusion saliency for harmed images — compare with LRP above
        # saliency_results = generate_clip_saliency(harmed_images, harmed_prompts, clip_ft)
        # fig, axes = plt.subplots(2, top_k, figsize=(2.5 * top_k, 5), squeeze=False)
        # fig.suptitle(
        #     f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — harmed: original | saliency",
        #     fontsize=10, fontweight="bold",
        # )
        # for col, (img_pil, sr) in enumerate(zip(harmed_images, saliency_results)):
        #     axes[0, col].imshow(img_pil)
        #     axes[0, col].axis("off")
        #     axes[0, col].set_title(harmed_prompts[col], fontsize=7)
        #     axes[1, col].imshow(sr["heatmap_rgb"])
        #     axes[1, col].axis("off")
        #     axes[1, col].set_title(f"sim={sr['similarity']:.2f}", fontsize=7)
        # plt.tight_layout()
        # sfpath = os.path.join(sae_dir, concept_extractor_name,ablation_hook.__qualname__ , f"ablation_harmed_saliency_group{gid}.png")
        # os.makedirs(os.path.dirname(sfpath), exist_ok=True)
        # plt.savefig(sfpath, bbox_inches="tight", dpi=120)
        # plt.close()
        # print(f"  Saved harmed saliency (group {gid}): {sfpath}")


@torch.no_grad()
def ablate_spurious_concepts_routesae(
    clip_ft, sae_model, test_ds,
    spurious_concept_indices,
    device, sae_dir,
    lambda_coefficient=1.0,
    vocab_names=None,
    concept_match_scores=None,
    concept_source_group=None,
    concept_mis_prevalence=None,
    concept_extractor_name=None,
    clip_mode=None,
    report_suffix="",
    editing_method="deactivation",
    P=None,
    write_report=True,
):
    """RouteSAE counterpart of ablate_spurious_concepts.

    Two editing methods, both via routesae_adapter, which handles RouteSAE's
    multi-layer hook registration/cleanup internally (it hooks each routed
    CLIP layer, not a single final-embedding hook like MSAE's ablation_hook):

    'deactivation' (default): scales the named concept latents by
        (1 - lambda_coefficient) via routesae_adapter.routesae_embeds.
    'projection': projects the routed residual stream onto the orthogonal
        complement of span(P) (a projection matrix built by
        build_projection_matrix() from the same concepts' decoder
        directions, via either the QR or pinv method) via
        routesae_adapter.routesae_embeds_projection -- see its docstring
        for how this differs from deactivation.

    Unlike ablate_spurious_concepts, this does not produce the
    "corrected-images" plots (those call into MSAE-specific per-pixel
    gradient/heatmap code with no RouteSAE analogue) -- report only.

    Saves: sae_dir/ablation_report_{extractor}__routesae_{deactivation|projection}{suffix}.txt
    """
    import clip as openai_clip
    from routesae_adapter import routesae_embeds, routesae_embeds_projection
    from PIL import Image as _PIL

    if editing_method not in ("deactivation", "projection"):
        raise ValueError(
            f"ablate_spurious_concepts_routesae: editing_method must be "
            f"'deactivation' or 'projection', got {editing_method!r}."
        )

    CLASS_PROMPTS = ["a photo of a landbird", "a photo of a waterbird"]
    tokens = openai_clip.tokenize(CLASS_PROMPTS).to(device)
    txt    = clip_ft.model.encode_text(tokens).float()
    txt    = txt / txt.norm(dim=-1, keepdim=True)

    group_ids   = np.array([s[3] for s in test_ds.samples])
    true_labels = np.array([s[1] for s in test_ds.samples])
    concept_idx = list(spurious_concept_indices)

    # Batched -- routesae_embeds/routesae_embeds_projection and the
    # RouteHook/RouteHookProjection masking underneath both already operate
    # on (batch, seq_len, ...) tensors (same as extract_routesae_representations's
    # batch_size=32 extraction), so this was never inherently per-image; it
    # just wasn't batched yet. Same BATCH_SIZE/MPS-flush pattern as
    # ablate_spurious_concepts's batching earlier this session.
    BATCH_SIZE = 64
    is_mps = torch.device(device).type == "mps"
    all_paths = [s[0] for s in test_ds.samples]

    orig_logits, ablated_logits = [], []
    for b in tqdm(range(0, len(all_paths), BATCH_SIZE), desc="  Original + ablated CLIP (RouteSAE)"):
        batch_paths = all_paths[b:b + BATCH_SIZE]
        imgs = torch.stack([
            clip_ft.preprocess(_PIL.open(p).convert("RGB")) for p in batch_paths
        ]).to(device)

        feat = clip_ft.model.encode_image(imgs).float()
        feat = feat / feat.norm(dim=-1, keepdim=True)
        orig_logits.append((feat @ txt.T).cpu())

        if editing_method == "projection":
            ablated_feat = routesae_embeds_projection(
                sae_model, clip_ft.model, imgs,
                P=P, lambda_coef=lambda_coefficient,
            ).float()
        else:
            ablated_feat = routesae_embeds(
                sae_model, clip_ft.model, imgs,
                concept_idx=concept_idx, lambda_coef=lambda_coefficient,
            ).float()
        ablated_feat = ablated_feat / ablated_feat.norm(dim=-1, keepdim=True)
        ablated_logits.append((ablated_feat @ txt.T).cpu())

        if is_mps and (b // BATCH_SIZE) % 10 == 0:
            torch.mps.synchronize()
            torch.mps.empty_cache()

    orig_logits    = torch.cat(orig_logits, dim=0)
    ablated_logits = torch.cat(ablated_logits, dim=0)
    orig_preds     = orig_logits.argmax(dim=1).numpy()
    ablated_preds  = ablated_logits.argmax(dim=1).numpy()

    if write_report:
        _write_ablation_report(
            group_ids=group_ids, true_labels=true_labels,
            orig_preds=orig_preds, ablated_preds=ablated_preds,
            concept_idx=concept_idx, vocab_names=vocab_names,
            concept_match_scores=concept_match_scores,
            concept_source_group=concept_source_group,
            concept_mis_prevalence=concept_mis_prevalence,
            sae_dir=sae_dir, concept_extractor_name=concept_extractor_name,
            hook_tag=f"routesae_{editing_method}", report_suffix=report_suffix,
            clip_mode=clip_mode,
        )
    return orig_preds, ablated_preds, f"routesae_{editing_method}"


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


def visualize_concepts_on_images(
    image_paths,
    clip_ft,
    sae_model,
    concept_ids,
    device,
    concept_match_scores=None,
    vocab_names=None,
    max_images=10,
    save_dir=None,
    group_name=None,
    sae_representations=None,
):
    """
    For each concept, plot all images in two horizontal rows:
      Row 0: original images, titled with their concept activation value
      Row 1: heatmap overlays
    Saves one file per concept to save_dir.
    """
    import matplotlib.pyplot as plt
    from overcomplete.visualization.plot_utils import show, interpolate_cv2, get_image_dimensions

    if isinstance(concept_ids, (int, np.integer)):
        concept_ids = [concept_ids]
    image_paths = list(image_paths)[:max_images]
    n_images    = len(image_paths)

    sae_reps = None
    if sae_representations is not None:
        sae_reps = sae_representations[:n_images]   # align to sliced paths

    for cid in concept_ids:
        cname = (
            vocab_names[concept_match_scores[:, cid].argmax()]
            if vocab_names is not None and concept_match_scores is not None
            else str(cid)
        )

        # Collect (img_pil, heatmap) per image
        panels = []
        for path in image_paths:
            result = concept_spatial_heatmap(
                image_path=path,
                clip_ft=clip_ft,
                sae_model=sae_model,
                sae_representations=sae_reps,
                concept_id=cid,
                device=device,
                concept_match_scores=concept_match_scores,
                vocab_names=vocab_names,
                return_only=True,
            )
            if result is not None:
                panels.append(result)

        if not panels:
            continue

        # Layout: 2 rows × n_images columns
        fig, axes = plt.subplots(2, n_images, figsize=(3 * n_images, 6))
        if n_images == 1:
            axes = [[axes[0]], [axes[1]]]

        title = f"Concept {cid}: {cname}"
        if group_name:
            title = f"{group_name} — {title}"
        fig.suptitle(title, fontsize=12)

        for col, (img_pil, heatmap) in enumerate(panels):
            width, height = get_image_dimensions(img_pil)
            heatmap_up    = interpolate_cv2(heatmap, (width, height))

            # Row 0: original image + activation value as title
            plt.sca(axes[0][col])
            show(img_pil)
            axes[0][col].axis("off")
            if sae_reps is not None:
                act_val = float(sae_reps[col, cid])
                axes[0][col].set_title(f"act={act_val:.3f}", fontsize=8)

            # Row 1: heatmap overlay
            plt.sca(axes[1][col])
            show(img_pil)
            show(heatmap_up, cmap="jet", alpha=0.5)
            axes[1][col].axis("off")

        plt.tight_layout()

        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            save_path = os.path.join(save_dir, f"concept_{cid}_{cname}.png")
            plt.savefig(save_path, dpi=150, bbox_inches="tight")
            print(f"  Saved heatmap grid → {save_path}")

        plt.close(fig)


def visualize_top_k_concept_images(
    image_paths,
    sae_representations,
    clip_ft,
    sae_model,
    concept_ids,
    device,
    k=10,
    concept_match_scores=None,
    vocab_names=None,
):
    """For each concept, find the k images with highest activation and show patch heatmaps."""
    from PIL import Image as _PIL

    image_paths = list(image_paths)
    vis = clip_ft.model.visual

    for cid in concept_ids:
        cname = (
            vocab_names[concept_match_scores[:, cid].argmax()]
            if vocab_names is not None and concept_match_scores is not None
            else str(cid)
        )

        # Select top-k images by global SAE activation for this concept
        top_indices = np.argsort(sae_representations[:, cid])[::-1][:k]
        selected_paths = [image_paths[i] for i in top_indices]
        images_pil = [_PIL.open(p).convert("RGB") for p in selected_paths]
        images_t   = torch.stack([clip_ft.preprocess(img) for img in images_pil]).to(device)

        # Extract patch tokens via hook on last transformer block
        _patch_tokens = []
        def _capture_hook(_m, _i, output):
            _patch_tokens.append(output.permute(1, 0, 2).detach())

        hook = vis.transformer.resblocks[-1].register_forward_hook(_capture_hook)
        with torch.no_grad():
            clip_ft.model.encode_image(images_t)
        hook.remove()

        patch_feats = _patch_tokens[0][:, 1:, :]   # drop CLS → [k, n_patches, d_hidden]

        # Project to CLIP output space to match SAE input dim
        with torch.no_grad():
            patch_feats = vis.ln_post(patch_feats)
            if vis.proj is not None:
                patch_feats = patch_feats @ vis.proj

        print(f"Concept {cid}: {cname}  (top-{k} images)")
        for i, path in zip(top_indices, selected_paths):
            concept_spatial_heatmap(
                image_path=path,
                clip_ft=clip_ft,
                sae_model=sae_model,
                sae_representations=sae_representations[i][np.newaxis, :],  # (1, d_sae)
                concept_id=cid,
                device=device,
                concept_match_scores=concept_match_scores,
                vocab_names=vocab_names,
            )


def concept_spatial_heatmap(
    image_path,
    clip_ft,
    sae_model,
    sae_representations,
    concept_id,
    device,
    concept_match_scores=None,
    vocab_names=None,
    return_only=False,
):
    """
    Attribute a concept's activation back to spatial image patches using gradients.

    Uses the same global-embedding SAE that was used for concept extraction:
      image → CLIP CLS embedding → SAE → concept activation
                     ↑ gradient flows back to patch tokens in last transformer block

    Returns heatmap of shape [grid_h, grid_w] (7×7 for ViT-B/32, 14×14 for ViT-B/16).
    """
    import math
    import matplotlib.pyplot as plt
    from PIL import Image as _PIL
    from routesae import RouteSAE

    # RouteSAE has no single global sae_model.encode() -- it's a per-layer/
    # per-patch residual-stream SAE with no single decoder to attribute a
    # concept's activation back through (see --model RouteSAE's help text in
    # parse_args()). This gradient-attribution heatmap only makes sense for
    # the MSAE family's single global embedding SAE, so skip cleanly for
    # RouteSAE; callers already treat None as "skipped".
    if isinstance(sae_model, RouteSAE):
        if not getattr(concept_spatial_heatmap, "_routesae_warned", False):
            print("  [info] Spatial concept heatmaps are not supported for RouteSAE — skipping.")
            concept_spatial_heatmap._routesae_warned = True
        return None

    # Skip images where the target concept never fired: the gradient of a zero
    # (ReLU/TopK-gated) activation is meaningless — the concept does not exist
    # in this image, so no heatmap is drawn. Callers treat None as "skipped".
    if sae_representations is not None:
        _acts = (
            sae_representations.cpu().numpy()
            if torch.is_tensor(sae_representations)
            else np.asarray(sae_representations)
        )
        if _acts.ndim == 2:
            _acts = _acts[0]
        if _acts[concept_id] <= 0:
            return None

    cname = (
        vocab_names[concept_match_scores[:, concept_id].argmax()]
        if vocab_names is not None and concept_match_scores is not None
        else str(concept_id)
    )

    img_pil = _PIL.open(image_path).convert("RGB")
    img_t   = clip_ft.preprocess(img_pil).unsqueeze(0).to(device)

    vis = clip_ft.model.visual

    # Hook the INPUT to the last transformer block (pre-hook).
    # Patch tokens here still influence the CLS output through the last block's
    # self-attention, so gradients from the concept activation flow back to them.
    # Hooking the OUTPUT instead gives zero patch-token gradients because encode_image
    # discards all positions except CLS (index 0) after the last block.
    from overcomplete.visualization.plot_utils import show, interpolate_cv2, get_image_dimensions

    # Hook conv1 (patch embedding layer) — its output is a leaf-adjacent node in the
    # computation graph (conv1 has learnable weights), so retain_grad() reliably works here.
    # Shape: [1, d_hidden, grid_h, grid_w]
    _patch_embeds = []
    def _embed_hook(_m, _i, output):
        output.retain_grad()
        _patch_embeds.append(output)

    hook = vis.conv1.register_forward_hook(_embed_hook)

    # torch.enable_grad() is required here: this function may be called from within
    # a @torch.no_grad() scope (e.g. ablate_spurious_concepts), which would prevent
    # retain_grad() from working and make backward() a no-op.
    with torch.enable_grad():
        clip_ft.model.zero_grad()
        feat = clip_ft.model.encode_image(img_t).float()   # [1, 512]
        hook.remove()
        # NOTE: no L2 normalization here — the SAE's preprocessing stats (mean,
        # scaling_factor) were fit on RAW encode_image features, and te_results /
        # extract_sae_representations encode raw features too. Normalizing would
        # attribute a different activation than the one used for concept selection
        # (and can kill the top-k gate entirely for TopK SAEs).

        # Run through SAE and get the specific concept's activation
        sparse_codes, _ = sae_model.encode(feat)           # [1, d_sae]
        concept_activation = sparse_codes[0, concept_id]

        concept_activation.backward()

    # grads: [d_hidden, grid_h, grid_w] — mean abs gradient across channels per patch
    grads   = _patch_embeds[0].grad[0]                 # [d_hidden, grid_h, grid_w]
    heatmap = grads.abs().mean(dim=0)                  # [grid_h, grid_w]
    heatmap = torch.nn.functional.relu(heatmap)
    heatmap = heatmap.detach().cpu().numpy()

    if heatmap.max() > 0:
        heatmap = heatmap / heatmap.max()

    # Upsample heatmap to image resolution using overcomplete's cv2-based interpolation
    width, height = get_image_dimensions(img_pil)
    heatmap_up    = interpolate_cv2(heatmap, (width, height))

    if return_only:
        return img_pil, heatmap

    # Plot: original | overlay
    plt.figure(figsize=(8, 4))
    plt.subplot(1, 2, 1)
    show(img_pil)
    plt.title("Original")

    plt.subplot(1, 2, 2)
    show(img_pil)
    show(heatmap_up, cmap="jet", alpha=0.5)
    plt.title(f"Concept {concept_id}: {cname}")

    plt.tight_layout()
    # plt.show()
    return heatmap


CLASS_NAMES = {0: "landbird", 1: "waterbird"}


def Save_candidate_report(M, Y_hat, M_centroid, M_knn, te_results, test_ds, save_path=None):
    """
    For every image flagged as a candidate (M[i] == True), save a row containing:
        img_idx       - index in the test set
        path          - image file path
        true_label    - ground-truth class (from test_ds.samples)
        y_hat         - CLIP pseudo-label
        centroid_pred - centroid-nearest-class prediction
        knn_pred      - k-NN majority-vote prediction
        in_centroid   - whether centroid flagged this image (M_centroid[i])
        in_knn        - whether k-NN flagged this image (M_knn[i])

    Saves a CSV to save_path (or prints if save_path is None).
    """
    import csv, io

    _cls = {0: "landbird", 1: "waterbird"}
    candidate_indices = M.nonzero(as_tuple=True)[0].tolist()
    image_paths       = te_results["image_paths"]
    y_hat_list        = Y_hat.cpu().tolist()
    m_cen_list        = M_centroid.cpu().tolist()
    m_knn_list        = M_knn.cpu().tolist()

    rows = []
    for i in candidate_indices:
        true_label   = test_ds.samples[i][1]
        y_hat_i      = y_hat_list[i]
        # centroid / knn predicted class: if they disagree with Y_hat, flip it
        centroid_pred = 1 - y_hat_i if m_cen_list[i] else y_hat_i
        knn_pred      = 1 - y_hat_i if m_knn_list[i] else y_hat_i
        rows.append({
            "img_idx":      i,
            "path":         image_paths[i],
            "true_label":   _cls.get(true_label, true_label),
            "y_hat":        _cls.get(y_hat_i, y_hat_i),
            "centroid_pred": _cls.get(centroid_pred, centroid_pred),
            "knn_pred":     _cls.get(knn_pred, knn_pred),
            "in_centroid":  bool(m_cen_list[i]),
            "in_knn":       bool(m_knn_list[i]),
        })

    n_mislabeled = sum(1 for r in rows if r["true_label"] != r["y_hat"])
    n_mislabeled_centroid = sum(1 for r in rows if r["true_label"] != r["centroid_pred"])
    n_mislabeled_knn = sum(1 for r in rows if r["true_label"] != r["knn_pred"])
    # Candidates where y_hat was already correct — these were flagged purely
    # because centroid/knn disagreed with a correct pseudo-label (false-positive
    # flags from the candidate-selection mechanism itself).
    n_correct_yhat_flagged = sum(1 for r in rows if r["true_label"] == r["y_hat"])
    fieldnames = ["img_idx", "path", "true_label", "y_hat",
                  "centroid_pred", "knn_pred", "in_centroid", "in_knn"]
    if rows:
        n = len(rows)
        summary = (
            f"candidates={n}  |  "
            f"true!=y_hat={n_mislabeled} ({100*n_mislabeled/n:.1f}%)  |  "
            f"true!=centroid={n_mislabeled_centroid} ({100*n_mislabeled_centroid/n:.1f}%)  |  "
            f"true!=knn={n_mislabeled_knn} ({100*n_mislabeled_knn/n:.1f}%)  |  "
            f"true==y_hat-but-flagged={n_correct_yhat_flagged} ({100*n_correct_yhat_flagged/n:.1f}%)"
        )
    else:
        summary = "candidates=0"

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", newline="") as f:
            f.write(f"# {summary}\n")
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        print(f"  Candidate report saved → {save_path}  ({summary})")
    else:
        buf = io.StringIO()
        buf.write(f"# {summary}\n")
        writer = csv.DictWriter(buf, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        print(buf.getvalue())

    return rows


def _count_concepts_chunk(args):
    """Worker for parallel concept counting (must be module-level for pickling)."""
    from collections import Counter
    chunk_items, candidate_set, pool_set = args
    counts = Counter()
    for img_idx, cids in chunk_items:
        if img_idx in candidate_set:
            for cid in cids:
                if cid in pool_set:
                    counts[cid] += 1
    return dict(counts)


def select_prevalent_concepts(M, per_image_pool, concept_pool,
                               prevalence_threshold=0.25, top_k=None,
                               n_cpu_workers=None):
    """
    Count how many candidate images each concept appears in, then filter.

    Two modes (mutually exclusive; top_k takes priority if both given):
      top_k mode       — keep the top_k concepts with the highest counts.
      threshold mode   — keep concepts present in ≥ prevalence_threshold
                         fraction of candidate images.

    Parameters
    ----------
    M                   : BoolTensor (N,) — candidate mask from candidate_selection
    per_image_pool      : dict {img_idx: list[cid]} — from Generate_Concept_Pool
    concept_pool        : list[int] — global concept pool
    prevalence_threshold: float — used when top_k is None (default 0.25)
    top_k               : int or None — if set, select top-k by count instead
    n_cpu_workers       : int or None — CPU workers; defaults to os.cpu_count()

    Returns
    -------
    candidate_concepts : list[int]  selected concept IDs (sorted by count desc
                         for top_k mode, sorted by ID for threshold mode)
    concept_counts     : dict {cid: int}  raw per-concept counts over candidates
    """
    from concurrent.futures import ProcessPoolExecutor
    from collections import Counter

    candidate_indices = set(M.nonzero(as_tuple=True)[0].tolist())
    concept_pool_set  = set(concept_pool)
    n_workers         = n_cpu_workers or os.cpu_count() or 4

    items       = list(per_image_pool.items())
    chunk_size  = max(1, (len(items) + n_workers - 1) // n_workers)
    chunks      = [items[i:i + chunk_size] for i in range(0, len(items), chunk_size)]
    chunk_args  = [(chunk, candidate_indices, concept_pool_set) for chunk in chunks]

    concept_counts = Counter()
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        for partial in pool.map(_count_concepts_chunk, chunk_args):
            concept_counts.update(partial)
    concept_counts = dict(concept_counts)

    n_cands = len(candidate_indices)
    if top_k is not None:
        # Top-K mode: pick the K concepts influential in the most candidate images
        ranked = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        candidate_concepts = [cid for cid, _ in ranked[:top_k]]
        print(
            f"  Candidate concepts: {len(candidate_concepts)} "
            f"(top-{top_k} by count over {n_cands} candidate images; "
            f"min count={concept_counts[candidate_concepts[-1]] if candidate_concepts else 0})"
        )
    else:
        # Threshold mode: keep concepts present in ≥ prevalence_threshold of candidates
        min_count = max(1, int(prevalence_threshold * n_cands))
        candidate_concepts = sorted(
            cid for cid, count in concept_counts.items() if count >= min_count
        )
        print(
            f"  Candidate concepts: {len(candidate_concepts)} "
            f"(present in ≥{prevalence_threshold*100:.0f}% of {n_cands} "
            f"candidate images, threshold={min_count})"
        )
    return candidate_concepts, concept_counts

@torch.no_grad()
def select_dialguided_concepts(
    te_results, ft_results, test_ds, ft_ds,
    concept_match_scores, vocab_names,
    sae_dir, clip_ft, sae_model, device,
    spurious_sim_threshold: float = 0.20,
    good_sim_threshold: float = 0.20,
    n_attrs: int = 20,
    discover_from_vocab: bool = False,
    use_image_grounding: bool = True,
    image_grounding_weight: float = 0.5,
    img_batch_size: int = 64,
    alpha: float = 0.75,
    sp_arribute_dir: str = None,
    top_n_concepts: int = 50,
):
    """
    Dialogue-guided spurious concept selection using a zero-shot (unbiased) CLIP.

    Two modes controlled by `discover_from_vocab`:

    discover_from_vocab=True  (default)
        Attribute sets are discovered from the SAE vocabulary (vocab_names).
        Two sub-modes controlled by `use_image_grounding`:

        use_image_grounding=True  (default)
            Score = (1 - w) * text_anchor_sim + w * image_persistence
            where text_anchor_sim  = cosine sim of vocab text emb to anchor dir
                  image_persistence = avg cosine sim of vocab text emb to
                                      class-specific image embeddings
                  w = image_grounding_weight  (default 0.5)
            → selects vocab terms that are BOTH semantically related to the
              background anchor AND visually persistent in the actual images.

        use_image_grounding=False
            Score = text_anchor_sim only (pure text-text matching).

        Anchors:
          landbird spurious  → "land background scenery"
          waterbird spurious → "water background scenery"
          good (bird body)   → "bird anatomy and body parts"

    discover_from_vocab=False  (backup / hardcoded)
        SPURIOUS_ATTRS and GOOD_ATTRS are predefined lists (see below).

    In both modes the final filtering is identical:
        A SAE concept is kept if
            max_sim(decoder_dir, spurious_attrs) >= spurious_sim_threshold
          AND
            max_sim(decoder_dir, good_attrs)     <  good_sim_threshold

    Hardcoded backup attribute dictionary
    --------------------------------------
    SPURIOUS landbird  : tree, forest, branch, bark, leaf, jungle, grass, shrub,
                         dirt, ground, soil, field, meadow, rock, stone, wood,
                         hillside, canopy, underbrush, woodland
    SPURIOUS waterbird : sea, ocean, water, wave, beach, coast, shore, lake,
                         river, dock, pier, boat, sky, cloud, horizon, surf,
                         tide, marsh, wetland, bay
    GOOD (shared)      : feather, wing, beak, bird head, tail, claw, eye,
                         plumage, breast, neck, body, foot, talon, crown, bill,
                         bird tail, bird wing, bird body, perch, avian
    """
    import json
    
    import clip as openai_clip
    # ── 1. Load zero-shot CLIP (base weights, not fine-tuned) ─────────────────
    if _hf_model_dir:
        _zs = CLIPZeroShot.load_offline(clip_ft.model_name, weights_dir=_hf_model_dir, device=str(device))
        zs_clip, zs_preprocess = _zs.model, _zs.preprocess
    else:
        zs_clip, zs_preprocess = openai_clip.load(clip_ft.model_name, device=device)
    zs_clip.eval()

    def _encode(phrases, batch_size=128):
        # MPS crashes on large token tensors; process in batches to stay safe
        all_embs = []
        for i in range(0, len(phrases), batch_size):
            batch = phrases[i : i + batch_size]
            tokens = openai_clip.tokenize(
                [f"a photo of {p}" for p in batch]
            ).to(device)
            emb = zs_clip.encode_text(tokens).float()
            emb = emb / emb.norm(dim=-1, keepdim=True)
            all_embs.append(emb)
        return torch.cat(all_embs, dim=0)  # (N, D)

    # ── 2a. Discover attribute sets from vocabulary ────────────────────────────
    if discover_from_vocab:
        # Encode every vocab term (these are human-readable SAE latent names)
        vocab_list = list(vocab_names)                     # V strings
        vocab_emb  = _encode(vocab_list)                   # (V, D)

        # Seed anchors: one representative phrase per category
        ANCHORS = {
            "landbird":  ["land background scenery",
                          "trees forest ground nature background",
                          "terrestrial landscape environment"],
            "waterbird": ["water background scenery",
                          "ocean sea lake shore aquatic background",
                          "marine coastal environment"],
            "good":      ["bird anatomy and body parts",
                          "avian features feathers wings beak",
                          "bird physical characteristics"],
        }

        # ── Encode class-specific images if image grounding is requested ─────
        if use_image_grounding:
            import PIL as _PIL
            # Separate test_ds samples by class label
            by_class: dict[int, list] = {}
            for path, label, *_ in test_ds.samples:
                by_class.setdefault(int(label), []).append(path)

            def _encode_images(paths):
                """Return mean-pooled, L2-normalised image embedding for a path list."""
                all_img_embs = []
                for i in range(0, len(paths), img_batch_size):
                    batch_paths = paths[i : i + img_batch_size]
                    imgs = torch.stack([
                        zs_preprocess(Image.open(p).convert("RGB"))
                        for p in batch_paths
                    ]).to(device)
                    img_emb = zs_clip.encode_image(imgs).float()
                    img_emb = img_emb / img_emb.norm(dim=-1, keepdim=True)
                    all_img_embs.append(img_emb)
                return torch.cat(all_img_embs, dim=0)   # (N_imgs, D)

            print("[dialguided] Encoding class images for grounding...")
            img_emb_by_class = {
                cls: _encode_images(paths)
                for cls, paths in by_class.items()
            }
            # All-class image embeddings for "good" attributes
            img_emb_all = torch.cat(list(img_emb_by_class.values()), dim=0)

        def _top_vocab(anchor_phrases, k, img_embs=None):
            """Score vocab terms by text-anchor sim, optionally reranked by image persistence."""
            anchor_emb = _encode(anchor_phrases)           # (A, D)
            anchor_dir = anchor_emb.mean(dim=0)
            anchor_dir = anchor_dir / anchor_dir.norm().clamp(min=1e-8)
            text_sim   = vocab_emb @ anchor_dir            # (V,) — text-text score

            if use_image_grounding and img_embs is not None:
                # image_persistence[v] = avg cosine sim of vocab_emb[v] with all class images
                # vocab_emb: (V, D),  img_embs: (N_imgs, D)
                # → (V, N_imgs) → mean over images → (V,)
                img_persist = (vocab_emb @ img_embs.T).mean(dim=-1)
                score = (1.0 - image_grounding_weight) * text_sim + \
                        image_grounding_weight * img_persist
            else:
                score = text_sim

            top_idx = score.topk(k).indices.tolist()
            return [vocab_list[i] for i in top_idx]

        _img_land  = img_emb_by_class.get(0) if use_image_grounding else None
        _img_water = img_emb_by_class.get(1) if use_image_grounding else None
        _img_all   = img_emb_all              if use_image_grounding else None

        land_attrs  = _top_vocab(ANCHORS["landbird"],  n_attrs, img_embs=_img_land)
        water_attrs = _top_vocab(ANCHORS["waterbird"], n_attrs, img_embs=_img_water)
        good_attrs  = _top_vocab(ANCHORS["good"],      n_attrs, img_embs=_img_all)

        SPURIOUS_ATTRS = {"landbird": land_attrs, "waterbird": water_attrs}
        GOOD_ATTRS     = good_attrs
        mode_tag = "vocab-discovery-image-grounded" if use_image_grounding else "vocab-discovery"

    # ── 2b. Hardcoded backup attribute sets ───────────────────────────────────
    else:
        # SPURIOUS_ATTRS = {
        #     "landbird": [
        #         "tree", "forest", "branch", "bark", "leaf", "jungle", "grass",
        #         "shrub", "dirt", "ground", "soil", "field", "meadow", "rock",
        #         "stone", "wood", "hillside", "canopy", "underbrush", "woodland", 
        #         "sky", "cloud", "sun", "mountain", "hill", "valley", "Horizon line"
        #         "bamboo", "cliff", "canyon", "desert", "savanna", "prairie", "forest floor",
        #         "rainforest", "tropical", "temperate", "deciduous", 
        #     ],
        #     "waterbird": [
        #         "sea", "ocean", "water", "wave", "beach", "coast", "shore",
        #         "lake", "river", "dock", "pier", "boat", "sky", "cloud",
        #         "horizon", "surf", "tide", "marsh", "wetland", "bay", "sun",
        #         "island", "reef", "lagoon", "estuary", "human", "bridge", 
        #         "fishing", "sailboat", "harbor", "lighthouse", "ship", "Shoreline",
                
        #     ],
        # }
        # GOOD_ATTRS = [
        #     "feather", "wing", "beak", "bird head", "tail", "claw", "eye",
        #     "plumage", "breast", "neck", "body", "foot", "talon", "crown",
        #     "bill", "bird tail", "bird wing", "bird body", "perch", "avian",
        # ]
        mode_tag = "hardcoded"
        SPURIOUS_ATTRS = _load_spurious_attrs_from_files(sp_arribute_dir,attr_type="spurious")
        GOOD_ATTRS = _load_spurious_attrs_from_files(sp_arribute_dir,attr_type="good")

    # ── 3. Encode chosen attribute sets ───────────────────────────────────────
    all_spurious = SPURIOUS_ATTRS["landbird"] + SPURIOUS_ATTRS["waterbird"]
    spurious_emb = _encode(all_spurious)   # (2*n_attrs, D)
    # good_emb     = _encode(GOOD_ATTRS)     # (n_attrs, D)
    
    # ── 4. DIAL-style scoring: s(f_j, a) per spurious attribute ──────────────
    #
    # For each attribute a:
    #   P_a / N_a  : partition of dataset via zero-shot CLIP classification
    #                ("a photo of <a>" vs "a photo without <a>")
    #   z_{i,j}    : SAE sparse activation of concept j for image i
    #   e_a        : text embedding of attribute a  (row of spurious_emb)
    #   f_j        : SAE decoder vector for concept j
    #
    #   s(f_j,a) = (mean_{i∈P_a}(z_{i,j}) − mean_{i∈N_a}(z_{i,j})) * CosSim(f_j, e_a)
    #
    #   K_a = smallest prefix of features sorted by s(f,a) DESC such that
    #         Σ|s_top_k| >= α * Σ|s_all|          (attribution mass threshold)
    #
    #   K = ∪_a K_a   (final candidate set)

    # 4.1 — Reuse pre-computed activations and reconstructions from te_results
    Z     = te_results["sae_representations"].float().cpu()   # (N, L)
    E_hat = te_results["sae_reconstructed"].float().cpu()     # (N, D)
    E_hat = E_hat / E_hat.norm(dim=-1, keepdim=True).clamp(min=1e-8)

    # 4.2 — SAE decoder directions F (L, D), L2-normalised.
    # MSAE's decoder already lives in CLIP's 512-d joint embedding space, so
    # its rows ARE concept directions in CLIP space directly. RouteSAE's
    # decoder instead lives in the 768-d residual stream at its routed
    # layers (n/4..3n/4) -- a different space entirely, not comparable to a
    # text embedding via raw cosine similarity. Reuse
    # routesae_naming.decoder_search_space(), which projects those residual-
    # stream directions through CLIP's own output head (ln_post + visual
    # projection) into the joint space -- the identical method that built
    # this run's concept_match .npy, so "concept j's direction" here stays
    # consistent with what vocab_names[j] (this concept's name) actually
    # means for this RouteSAE.
    from routesae import RouteSAE
    if isinstance(sae_model, RouteSAE):
        from routesae_naming import decoder_search_space
        F = decoder_search_space(sae_model, clip_ft.model, patch_diff=True).detach().float().cpu()  # (L, D)
    else:
        _ae   = sae_model.model
        dec   = _ae.decoder if _ae.decoder is not None else _ae.encoder.t()  #  the learned dictionary of concept directions in CLIP space.
        F     = dec.detach().float().cpu()                         # (L, D)
    F_norm = F / F.norm(dim=-1, keepdim=True).clamp(min=1e-8) # (L, D).  F_j : concept j's decoder direction in CLIP space.

    # 4.3 — Encode positive/negative prompts for every spurious attribute
    pos_prompts = [f"a photo of {a}"         for a in all_spurious]
    neg_prompts = [f"a photo without {a}"    for a in all_spurious]
    pos_emb = _encode(pos_prompts).cpu()   # (A, D)
    neg_emb = _encode(neg_prompts).cpu()   # (A, D)
    spur_emb_cpu = spurious_emb.cpu()      # (A, D) — attribute text embeddings

    # 4.4 — Score and select K_a for each attribute, accumulate K
    K: set[int] = set()
    for a_idx, a_name in enumerate(all_spurious):
        e_a   = spur_emb_cpu[a_idx]          # (D,)
        p_emb = pos_emb[a_idx]               # (D,)
        n_emb = neg_emb[a_idx]               # (D,)

        # Partition: P_a = images where sim(ê_i, pos) > sim(ê_i, neg)
        sim_pos  = E_hat @ p_emb             # (N,)
        sim_neg  = E_hat @ n_emb             # (N,)
        P_mask   = sim_pos > sim_neg         # (N,) bool
        N_mask   = ~P_mask

        if P_mask.sum() == 0 or N_mask.sum() == 0:
            continue

        # Mean activations over P_a and N_a → (L,)
        mu_P = Z[P_mask].mean(dim=0)
        mu_N = Z[N_mask].mean(dim=0)

        # CosSim(f_j, e_a) → (L,)
        cos_sim = F_norm @ e_a               # (L,)

        # s(f_j, a) → (L,)
        s = (mu_P - mu_N) * cos_sim

        # Select K_a: smallest prefix covering fraction α of total |s| mass (not sure if it should be absolute value or count, and what does it mean if it should be count, but I don't think it should be absolute value)
        abs_s       = s.abs()
        total_mass  = abs_s.sum()
        if total_mass == 0:
            continue
        _, sorted_idx = abs_s.sort(descending=True)
        cumsum = abs_s[sorted_idx].cumsum(0)
        k = int((cumsum < alpha * total_mass).sum().item()) + 1
        K_a = sorted_idx[:k].tolist()
        K.update(K_a)

        print(f"[dialguided] attr '{a_name}': |P_a|={P_mask.sum()}, "
              f"|N_a|={N_mask.sum()}, |K_a|={len(K_a)}")

    candidate_concepts = sorted(K)
    print(f"[dialguided] Total K = union of K_a: {len(candidate_concepts)} concepts")

    # ── 5. Log and save ───────────────────────────────────────────────────────
    attr_dict = {"mode": mode_tag, "spurious": SPURIOUS_ATTRS, "good": GOOD_ATTRS}
    print(f"[dialguided/{mode_tag}] spurious attrs: {len(all_spurious)}  "
          f"good attrs: {len(GOOD_ATTRS)}")
    print(f"[dialguided/{mode_tag}] SPURIOUS landbird : {SPURIOUS_ATTRS['landbird']}")
    print(f"[dialguided/{mode_tag}] SPURIOUS waterbird: {SPURIOUS_ATTRS['waterbird']}")
    print(f"[dialguided/{mode_tag}] GOOD              : {GOOD_ATTRS}")
    print(f"[dialguided/{mode_tag}] SAE concepts selected: {len(candidate_concepts)}")

    attr_path = os.path.join(sae_dir, f"dialguided_attribute_dict_{mode_tag}.json")
    with open(attr_path, "w") as f:
        json.dump(attr_dict, f, indent=2)
    print(f"[dialguided/{mode_tag}] Attribute dict saved → {attr_path}")

    return candidate_concepts


############## Deonie candidate conepts #########
@torch.no_grad()
def denois_candidate_concepts(candidate_concepts, sae_model, beta=75.0, percentile=0.75):
    """
    Filter candidate concepts by cosine similarity to the group mean decoder direction.

    For each concept, retrieve its decoder direction (the row of the SAE decoder
    matrix). Concepts whose direction is far from the group mean are considered
    noise and zeroed out.

    Steps:
        1. Retrieve decoder vector f_j for each concept ID  →  (K, d)
        2. Compute mean direction  m = mean(f_j over all K)  →  (1, d)
        3. Score each concept:  s[j] = beta * cosine_similarity(f_j, m)
           beta scales the scores before softmax to sharpen the distribution.
        4. Convert scores to weights:  w = softmax(s)  →  (K,)  sums to 1
        5. Zero out the bottom `percentile` fraction:
               w[j] = 0  if  w[j] < quantile(w, percentile)
           e.g. percentile=0.75 keeps only the top 25% by weight.

    Parameters
    ----------
    candidate_concepts : list[int]   concept IDs to evaluate
    sae_model          : SAE         model whose decoder rows are concept directions
    beta               : float       sharpening factor for softmax (default 75.0)
    percentile         : float       fraction of concepts to zero out, e.g. 0.75
                                     zeros the bottom 75% leaving top 25% (default 0.75)

    Returns
    -------
    filtered_concepts : list[int]         concept IDs with w[j] > 0 after thresholding
    w                 : torch.Tensor (K,) full weight vector; zeroed entries are filtered
    """
    if len(candidate_concepts) == 0:
        return [], torch.tensor([])

    # decoder rows are concept directions: (n_latents, n_inputs)
    # sae_model.model is the inner Autoencoder; .decoder is the nn.Parameter
    _ae  = sae_model.model
    dec  = _ae.decoder if _ae.decoder is not None else _ae.encoder.t()
    concept_vecs = dec[candidate_concepts].detach().float()          # (K, d)

    m = concept_vecs.mean(dim=0, keepdim=True)                       # (1, d)
    s = beta * F.cosine_similarity(concept_vecs, m.expand_as(concept_vecs), dim=-1)  # (K,)
    w = torch.softmax(s, dim=0)                                      # (K,)
    w[w < torch.quantile(w, percentile)] = 0.0

    filtered_concepts = [cid for cid, wi in zip(candidate_concepts, w.tolist()) if wi > 0]

    return filtered_concepts, w


class _ConceptActivationModel(torch.nn.Module):
    """Images (B, 3, H, W) → pre-activations (B, K) of K selected SAE latents.

    Dispatches on SAE architecture: MSAE reads CLIP's final joint embedding,
    RouteSAE reads the routed intermediate residual stream (see
    forward_routesae).

    Feeds the SAE raw CLIP features (no L2 normalization) — the same
    convention used everywhere else in this pipeline; SAE.preprocess()
    applies its own mean-centering/scaling.

    Maximizes the concepts' PRE-activations rather than the ReLU/TopK-gated
    latents: a gated latent has exactly zero gradient whenever the concept
    is inactive, which is always the case at the start of MACO's
    optimization from a random-phase image, so the loss would never move.

    Module-level (rather than nested in visualize_candidate_concepts_with_maco)
    so tests can import and exercise it directly; it closes over nothing from
    that function.
    """

    def __init__(self, clip_ft, sae_model, concept_ids):
        super().__init__()
        self.clip_ft = clip_ft
        self.sae_model = sae_model
        self.concept_ids = [int(c) for c in concept_ids]

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        from routesae import RouteSAE
        if isinstance(self.sae_model, RouteSAE):
            return self.forward_routesae(images)
        feats = self.clip_ft.model.encode_image(images).float()
        inner = getattr(self.sae_model, "model", None)
        if inner is not None and hasattr(inner, "encode_pre_act"):
            x = self.sae_model.preprocess(feats)
            if hasattr(inner, "preprocess"):
                x, _ = inner.preprocess(x)
            pre_acts = inner.encode_pre_act(x)
        else:
            # Fallback for SAE objects without an inner Autoencoder:
            # use the ungated dense latents.
            _, pre_acts = self.sae_model.encode(feats)
        return pre_acts[:, self.concept_ids]

    def forward_routesae(self, images: torch.Tensor) -> torch.Tensor:
        """RouteSAE path: CLIP's routed intermediate residual stream, not its
        final embedding.

        Uses SOFT routing (unlike inference/ablation, which use hard): hard
        routing's layer choice is a discrete argmax, and because maximizing
        pre_acts also drives max_weights up for the currently-selected layer,
        MACO would lock into whatever routing the random-phase init happened
        to pick and never migrate -- potentially maximizing the concept on the
        wrong layer's activations entirely. Soft routing is smooth and, on
        this router, lands in nearly the same place anyway (cos ≈ 0.93 vs the
        hard-routed input on real images).

        Caveat: soft routing keeps ~1.6x the magnitude of hard routing (its
        weights sum to 1 rather than scaling one layer down by max_weights),
        and pre_acts is linear in its input, so these activation VALUES run
        hotter than the same concept's elsewhere in the pipeline. Fine for
        maximization (scale doesn't move the argmax image); don't compare the
        numbers across paths.
        """
        from routesae import clip_layer_stack, pre_process
        stack = clip_layer_stack(self.clip_ft.model, images,
                                 self.sae_model.n_layers, enable_grad=True)  # (B, T, routed_layers, H)
        x = pre_process(stack)[0]
        rw = self.sae_model.get_router_weights(x, 'sum')
        routed_x = self.sae_model.get_sae_input(x, rw, 'soft')[1]            # (B, T, H)
        pre_acts = self.sae_model.sae.pre_acts(routed_x)                     # (B, T, latent)
        # Pool patches -> image-level, matching routesae_adapter.image_concepts'
        # defaults (max, CLS included). Without this the concept index below
        # would hit the patch axis instead of the latent axis.
        pre_acts = pre_acts.max(dim=1)[0]                                    # (B, latent)
        return pre_acts[:, self.concept_ids]


def visualize_candidate_concepts_with_maco(
    candidate_concepts,
    sae_model,
    clip_ft,
    test_ds,
    sae_dir,
    concept_match_scores=None,
    vocab_names=None,
    num_steps=128,
    num_crops=16,
    reference_samples=64,
    image_size=224,
    device=None,
    concept_batch=4,
    early_stop_patience=0,
    early_stop_delta=1e-3,
    shard="0/1",
    concept_names=None,
    subfolder=None,
):
    """Synthesize one MACO visualization per concept by maximizing its SAE latent.

    This ports the core MACO optimization loop from gatheluck/MACO and targets
    a model whose outputs are the selected SAE concept pre-activations.

    Concepts are optimized jointly in chunks of `concept_batch`: each step runs
    one CLIP forward/backward on a batch of concept_batch × num_crops crops
    with an additive per-concept loss, which is mathematically identical to
    optimizing each concept alone but ~concept_batch× faster.
    """
    import matplotlib.pyplot as plt
    from torch.utils.data import DataLoader, Dataset

    if not candidate_concepts:
        return []

    # Shard selection for cluster-level parallelism (SLURM job arrays):
    # "i/N" keeps concepts at positions i, i+N, i+2N, … so N array tasks cover
    # all concepts with no overlap. The default "0/1" keeps everything.
    if shard and shard != "0/1":
        shard_idx, shard_n = (int(v) for v in shard.split("/"))
        candidate_concepts = [
            c for j, c in enumerate(candidate_concepts) if j % shard_n == shard_idx
        ]
        if not candidate_concepts:
            return []

    if device is None:
        device = next(clip_ft.model.parameters()).device

    # MPS reroute: several ops MACO needs are missing on MPS and the measured
    # throughput there (per-step CPU sync for the crop, small-batch ViT
    # backward) is ~60x slower than plain CPU. CUDA runs natively.
    orig_model_device = next(clip_ft.model.parameters()).device
    if torch.device(device).type == "mps":
        print("  MACO: MPS device detected — running this stage on CPU "
              "(fully supported and faster than MPS for this workload).")
        device = "cpu"
        clip_ft.model.to(device)
        sae_model.to(device)

    def _safe_name(name: str) -> str:
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_") or "concept"

    def _concept_name(cid: int) -> str:
        # Precomputed names (used by parallel workers to avoid shipping the full
        # concept_match_scores matrix) take priority.
        if concept_names is not None and cid in concept_names:
            return concept_names[cid]
        if vocab_names is not None and concept_match_scores is not None:
            return vocab_names[concept_match_scores[:, cid].argmax()]
        return str(cid)

    class _RawImageDataset(Dataset):
        def __init__(self, image_paths):
            self.image_paths = list(image_paths)
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
            ])

        def __len__(self):
            return len(self.image_paths)

        def __getitem__(self, idx):
            path = self.image_paths[idx]
            img = Image.open(path).convert("RGB")
            return self.transform(img), 0

    def _normalize_transform():
        preprocess = getattr(clip_ft, "preprocess", None)
        if isinstance(preprocess, transforms.Compose):
            for t in preprocess.transforms:
                if isinstance(t, transforms.Normalize):
                    return t
        return transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),    ## default CLIP normalization values, hardcoded in their clip.load() function
        )

    def _compute_average_magnitude(dataloader):
        first_batch, _ = next(iter(dataloader))
        height, width = first_batch.shape[-2:]
        total_magnitude = torch.zeros(3, height, width, device=device)
        total_samples = 0
        with torch.no_grad():
            for images, _ in tqdm(dataloader, desc="  MACO magnitude"):
                images = images.to(device)
                fft_images = torch.fft.fft2(images, norm="backward")
                total_magnitude += torch.abs(fft_images).sum(dim=0)
                total_samples += images.size(0)
        if total_samples == 0:
            raise ValueError("Cannot compute MACO magnitude from an empty dataloader.")
        return (total_magnitude / total_samples).cpu()

    def _normalize(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
        return (x - x.min()) / (x.max() - x.min() + eps)

    def _recorrelate_colors(images: torch.Tensor) -> torch.Tensor:
        """Accepts (3, H, W) or a batch (K, 3, H, W)."""
        squeeze = images.ndim == 3
        if squeeze:
            images = images[None]
        assert images.ndim == 4 and images.size(1) == 3
        imagenet_color_correlation = torch.tensor(
            [
                [0.56282854, 0.58447580, 0.58447580],
                [0.19482528, 0.00000000, -0.19482528],
                [0.04329450, -0.10823626, 0.06494176],
            ],
            dtype=images.dtype,
            device=images.device,
        )
        images = torch.einsum("kchw,cd->kdhw", images, imagenet_color_correlation)
        return images[0] if squeeze else images

    def _normalize_alpha(alpha: torch.Tensor, percentile: float = 80.0) -> torch.Tensor:
        assert alpha.ndim == 3 and alpha.size(0) == 3
        alpha_mean = torch.mean(alpha, dim=0, keepdim=True)
        alpha_clamped = torch.clamp(alpha_mean, max=torch.quantile(alpha_mean, percentile / 100.0))
        return alpha_clamped / (alpha_clamped.max() + 1e-8)

    def _fourier_to_image(unnormalized_average_magnitude: torch.Tensor, phase: torch.Tensor) -> torch.Tensor:
        """Phase (3, H, W) or batched (K, 3, H, W) → image(s) of the same shape.

        Standardization is per concept image, matching the reference's
        single-image behaviour exactly for each batch element.
        """
        eps = 1e-5
        squeeze = phase.ndim == 3
        if squeeze:
            phase = phase[None]
        dims = (1, 2, 3)
        phase = phase - phase.mean(dim=dims, keepdim=True)
        phase = phase / (phase.std(dim=dims, keepdim=True) + eps)
        magnitude = unnormalized_average_magnitude
        if magnitude.ndim == 3:
            magnitude = magnitude[None]
        spectrum = torch.polar(magnitude.expand_as(phase), phase)
        image = torch.real(torch.fft.ifft2(spectrum, norm="backward"))
        image = image - image.mean(dim=dims, keepdim=True)
        image = image / (image.std(dim=dims, keepdim=True) + eps)
        image = _recorrelate_colors(image)
        image = torch.nn.functional.sigmoid(image)
        return image[0] if squeeze else image

    def _run_maco(model, num_concepts, unnormalized_average_magnitude, normalize_transform):
        """Jointly optimize `num_concepts` phase tensors against `model`,
        whose output column k must be concept k's activation.

        Returns (phases, alphas), each of shape (num_concepts, 3, H, W).
        """
        model = model.to(device).eval()
        device_type = torch.device(device).type
        # torch.compile pays off on CUDA where the identical graph is reused
        # every step; on CPU the compile overhead outweighs the gain.
        if device_type == "cuda":
            try:
                model = torch.compile(model)
            except Exception as e:
                print(f"  MACO: torch.compile unavailable ({e}); running eager.")

        unnormalized_average_magnitude = unnormalized_average_magnitude.to(device)
        phase = (2 * torch.pi * torch.rand(
            (num_concepts,) + tuple(unnormalized_average_magnitude.shape), device=device)
        ) - torch.pi
        phase.requires_grad = True
        alpha = torch.zeros_like(phase.detach())
        optimizer = torch.optim.NAdam([phase], lr=1.0)

        crop_transforms = [
            _MacoRandomResizedCrop((image_size, image_size), average_crop_size=b)
            for b in torch.linspace(0.5, 0.05, steps=num_steps, dtype=torch.float32)
        ]

        def get_crop_transform(step):
            return crop_transforms[step]

        # Built on CPU: only scalar values are read per step, and logspace is
        # not implemented on MPS.
        noise_stds = torch.logspace(0, -4, steps=num_steps, dtype=torch.float32)

        # Column index of the target concept for every crop in the joint batch:
        # crops [k*num_crops, (k+1)*num_crops) belong to concept k.
        best_activation = float("-inf")
        stale_steps = 0

        for step in tqdm(range(num_steps), desc="  MACO optimize"):
            optimizer.zero_grad()
            x_n = _fourier_to_image(unnormalized_average_magnitude, phase)  # (K, 3, H, W)
            normalized_x_n = normalize_transform(x_n)
            normalized_x_n.retain_grad()

            batch = normalized_x_n.repeat_interleave(num_crops, dim=0)  # (K*crops, 3, H, W)
            cropped_x_n = get_crop_transform(step)(batch)
            noise_std = float(noise_stds[step].item())
            cropped_x_n = cropped_x_n + torch.randn_like(cropped_x_n) * noise_std
            cropped_x_n = cropped_x_n + torch.rand_like(cropped_x_n) * noise_std - (noise_std / 2.0)

            # fp16 autocast on CUDA: ~2-3x faster CLIP forward/backward; the
            # phase/FFT math above stays in fp32.
            if device_type == "cuda":
                with torch.autocast("cuda", dtype=torch.float16):
                    acts = model(cropped_x_n)
            else:
                acts = model(cropped_x_n)

            acts = acts.float()
            row = torch.arange(acts.size(0), device=acts.device)
            col = torch.arange(num_concepts, device=acts.device).repeat_interleave(num_crops)
            target = acts[row, col]
            # Additive per-concept loss: each phase only receives gradient
            # from its own crops, exactly as in independent runs.
            loss = -target.mean()
            loss.backward()

            alpha += normalized_x_n.grad.abs()
            optimizer.step()

            # Early stopping (early_stop_patience=0 disables it): stop once the
            # mean concept activation has not improved by early_stop_delta for
            # `early_stop_patience` consecutive steps.
            if early_stop_patience > 0:
                current_activation = -float(loss.item())
                if current_activation > best_activation + early_stop_delta:
                    best_activation = current_activation
                    stale_steps = 0
                else:
                    stale_steps += 1
                    if stale_steps >= early_stop_patience:
                        print(f"  MACO: early stop at step {step + 1}/{num_steps} "
                              f"(no improvement for {early_stop_patience} steps).")
                        break

        return phase.detach().cpu(), alpha.detach().cpu()

    class _MacoRandomResizedCrop:
        def __init__(self, output_size, average_crop_size=0.25, center_std=0.15, delta_std=0.05, min_crop=0.05, max_crop=1.0):
            self.output_size = output_size
            self.average_crop_size = average_crop_size
            self.center_std = center_std
            self.delta_std = delta_std
            self.min_crop = min_crop
            self.max_crop = max_crop

        def __call__(self, images: torch.Tensor) -> torch.Tensor:
            # grid_sampler_2d_backward is not implemented on MPS — run the
            # crop on CPU there and move the result back; autograd routes the
            # backward of this op through CPU while CLIP stays on the GPU.
            if images.device.type == "mps":
                return self(images.cpu()).to(images.device)

            B, C, _, _ = images.shape
            device_local = images.device
            target_h, target_w = self.output_size

            center_x = 0.5 + torch.randn(B, device=device_local) * self.center_std
            center_y = 0.5 + torch.randn(B, device=device_local) * self.center_std
            delta = self.average_crop_size + torch.randn(B, device=device_local) * self.delta_std
            delta = delta.clamp(self.min_crop, self.max_crop)

            theta = torch.zeros(B, 2, 3, device=device_local)
            theta[:, 0, 0] = delta
            theta[:, 1, 1] = delta
            theta[:, 0, 2] = 2 * center_x - 1
            theta[:, 1, 2] = 2 * center_y - 1
            grid = torch.nn.functional.affine_grid(theta, size=(B, C, target_h, target_w), align_corners=True)
            return torch.nn.functional.grid_sample(images, grid, mode="bilinear", padding_mode="zeros", align_corners=True)

    # Output dir — optionally a per-method subfolder so concepts found/removed by
    # different concept-finding methods don't overwrite each other.
    out_dir = (os.path.join(sae_dir, "maco_concepts", subfolder) if subfolder
               else os.path.join(sae_dir, "maco_concepts"))
    os.makedirs(out_dir, exist_ok=True)

    # Resume support: skip any concept whose PNG already exists, so an interrupted
    # run continues instead of regenerating everything.
    existing_paths, pending = [], []
    for cid in candidate_concepts:
        png = os.path.join(out_dir, f"concept_{cid}_{_safe_name(_concept_name(cid))}.png")
        if os.path.exists(png):
            existing_paths.append(png)
        else:
            pending.append(cid)
    if existing_paths:
        print(f"  MACO: skipping {len(existing_paths)} already-rendered concept(s) in {out_dir}")
    candidate_concepts = pending
    if not candidate_concepts:
        print("  MACO: all requested concepts already rendered — nothing to do.")
        return existing_paths

    image_paths = [s[0] for s in test_ds.samples[:max(1, min(reference_samples, len(test_ds.samples)))]]
    ref_loader = DataLoader(_RawImageDataset(image_paths), batch_size=16, shuffle=False, num_workers=0)
    avg_magnitude = _compute_average_magnitude(ref_loader)
    normalize_transform = _normalize_transform()

    # Freeze CLIP/SAE weights during MACO: the phase tensor is the only leaf
    # being optimized, and letting backward accumulate .grad buffers in the
    # model weights over num_steps × num_crops passes only wastes memory.
    frozen = []
    for module in (clip_ft.model, sae_model):
        if hasattr(module, "parameters"):
            for p in module.parameters():
                frozen.append((p, p.requires_grad))
                p.requires_grad_(False)

    saved_paths = []
    try:
        # Concepts are optimized jointly in chunks: each step runs one CLIP
        # forward/backward serving the whole chunk (batch = chunk × num_crops).
        chunk_size = max(1, int(concept_batch))
        for start in range(0, len(candidate_concepts), chunk_size):
            chunk = list(candidate_concepts[start:start + chunk_size])
            model = _ConceptActivationModel(clip_ft, sae_model, chunk)
            phases, alphas = _run_maco(model, len(chunk), avg_magnitude, normalize_transform)

            for k, cid in enumerate(chunk):
                cname = _safe_name(_concept_name(cid))
                x = _fourier_to_image(avg_magnitude, phases[k])
                alpha_normalized = _normalize_alpha(alphas[k])
                x_alpha = _normalize(x * alpha_normalized)

                # Only "Image x Alpha" is shown — MACO image / Alpha panels commented
                # out below, not deleted, in case we want the 3-panel view back.
                # fig, axes = plt.subplots(1, 3, figsize=(9, 3))
                # panels = [x, alpha_normalized.repeat(3, 1, 1), x_alpha]
                # titles = ["MACO image", "Alpha", "Image × Alpha"]
                # for ax, panel, title in zip(axes, panels, titles):
                #     ax.imshow(panel.permute(1, 2, 0).clamp(0, 1))
                #     ax.axis("off")
                #     ax.set_title(title, fontsize=8)
                fig, ax = plt.subplots(1, 1, figsize=(3, 3))
                ax.imshow(x_alpha.permute(1, 2, 0).clamp(0, 1))
                ax.axis("off")
                fig.suptitle(f"Concept {cid}: {cname}", fontsize=11)
                plt.tight_layout()

                out_path = os.path.join(out_dir, f"concept_{cid}_{cname}.png")
                fig.savefig(out_path, bbox_inches="tight", dpi=150)
                plt.close(fig)
                saved_paths.append(out_path)
                print(f"  Saved MACO visualization: {out_path}")
    finally:
        for p, rg in frozen:
            p.requires_grad_(rg)
        if orig_model_device != next(clip_ft.model.parameters()).device:
            clip_ft.model.to(orig_model_device)
            sae_model.to(orig_model_device)

    return existing_paths + saved_paths


def _maco_parallel_worker(payload):
    """Worker process for parallel MACO. Reloads CLIP+SAE from disk (so nothing
    heavy is pickled across the process boundary) and runs the serial MACO
    visualization on its own shard of concepts. Returns the saved PNG paths."""
    import os as _os
    # Pin this worker to one GPU (multi-GPU nodes) BEFORE any CUDA context exists.
    if payload["gpu"] is not None:
        _os.environ["CUDA_VISIBLE_DEVICES"] = str(payload["gpu"])
    import types
    import torch as _torch
    if payload["threads"]:
        _torch.set_num_threads(payload["threads"])

    from clip_zero_shot import CLIPZeroShot

    ml = payload["model_load"]
    device = payload["device"]
    # Spawned workers start fresh and don't inherit the module global that main()
    # sets from --hf_model_dir. Restore it so offline from_pretrained() lookups on
    # the server resolve exactly as in the parent process.
    global _hf_model_dir
    _hf_model_dir = ml["hf_model_dir"]
    # Mirror main()'s CLIP construction exactly for each mode.
    if ml["mode"] == "zeroshot":
        clip_ft = CLIPZeroShot(model_name=ml["clip_model"], device=device)
    else:
        clip_ft = CLIPZeroShot.load_model(ml["model_pt"], device=device)
    clip_ft.model.eval()

    if ml.get("sae_kind") == "routesae":
        from routesae_adapter import load_routesae_for_clip
        # Mirror main()'s RouteSAE setup: routesae.py upcasts intermediate
        # activations to fp32, so a fp16 CLIP (which load_model can produce on
        # a non-CPU device) crashes at the first matmul against a fp16 weight.
        clip_ft.model = clip_ft.model.float()
        sae_model = load_routesae_for_clip(ml["sae_path"], device=device,
                                           k=ml.get("routesae_k", 32)).eval()
    else:
        from msae.sae import SAE
        sae_model = SAE(ml["sae_path"]).to(device).eval()

    # Minimal stand-in for the dataset: the viz only reads test_ds.samples[i][0].
    test_ds = types.SimpleNamespace(samples=payload["samples"])

    return visualize_candidate_concepts_with_maco(
        candidate_concepts=payload["shard"],
        sae_model=sae_model,
        clip_ft=clip_ft,
        test_ds=test_ds,
        sae_dir=payload["sae_dir"],
        concept_names=payload["concept_names"],
        device=device,
        **payload["maco_kwargs"],
    )


def top_concepts_by_activation(concepts, sae_representations, max_n, label=""):
    """Trim `concepts` to the `max_n` most active, ranked by peak activation.

    Per-concept visualization stages (MACO renders, top-image montages) cost
    one unit of work per concept, so they don't survive a concept-finding
    method that returns thousands -- dialguided has returned 12132 concepts on
    RouteSAE, 74% of the whole dictionary. Ranking by each concept's strongest
    activation anywhere in the split keeps the ones that most visibly fire.

    Only trims what gets VISUALIZED. Ablation runs on the full candidate list
    upstream of this, so reported accuracies are unaffected.

    Parameters
    ----------
    concepts            : list[int]  candidate concept IDs
    sae_representations : Tensor|ndarray (N_images, n_latents)
    max_n               : int | None  cap; None or <=0 disables trimming
    label               : str         name for the log line ("MACO", ...)

    Returns
    -------
    list[int] -- concepts unchanged if already within the cap, else the top
    max_n ordered most-active first.
    """
    if not max_n or max_n <= 0 or len(concepts) <= max_n:
        return list(concepts)

    reps = sae_representations
    reps_np = reps.cpu().numpy() if torch.is_tensor(reps) else np.asarray(reps)
    peak_activation = reps_np[:, concepts].max(axis=0)          # (K,)
    top_order = np.argsort(-peak_activation)[:max_n]
    trimmed = [concepts[i] for i in top_order]

    tag = f"{label} " if label else ""
    print(f"  {tag}concept cap: {len(concepts)} candidates exceeds {max_n} — "
          f"keeping the {max_n} most active (by peak activation). "
          f"Ablation still used all {len(concepts)}.")
    return trimmed


def save_top_ft_images_per_concept(
    candidate_concepts, ft_results, sae_dir, concept_extractor_name,
    clip_ft, sae_model, device,
    vocab_names=None, concept_match_scores=None, top_k=5,
):
    """For each candidate concept, save a 2-row montage of the top-k fine-tune-train
    images by that concept's SAE activation — the real-image counterpart to the
    synthetic MACO image:
        row 1: the top-k images (titled with activation),
        row 2: the same images with the concept's spatial heatmap overlaid
               (gradient attribution via concept_spatial_heatmap), highlighting
               the regions that most drive the concept.
    Written under:
        <sae_dir>/<concept_extractor_name>/top_ft_images/concept_<id>_<name>.png
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from PIL import Image as _PIL

    out_dir = os.path.join(sae_dir, concept_extractor_name, "top_ft_images")
    os.makedirs(out_dir, exist_ok=True)

    sae_reps = ft_results["sae_representations"]

    def _name(cid):
        if vocab_names is not None and concept_match_scores is not None:
            return vocab_names[concept_match_scores[:, cid].argmax()]
        return str(cid)

    def _safe(s):
        return re.sub(r"[^a-zA-Z0-9._-]+", "_", str(s)).strip("_") or "concept"

    saved = []
    for cid in candidate_concepts:
        hits = find_top_concept_images(ft_results, top_k=top_k, concept_index=cid)
        cname = _name(cid)
        n = len(hits)
        fig, axes = plt.subplots(2, n, figsize=(3 * n, 6.6), squeeze=False)
        for col, h in enumerate(hits):
            idx = h["image_index"]
            try:
                img_pil = _PIL.open(h["image_path"]).convert("RGB")
            except Exception:
                img_pil = None

            # Row 1 — raw image
            ax0 = axes[0][col]
            if img_pil is not None:
                ax0.imshow(img_pil)
            ax0.axis("off")
            ax0.set_title(f"act {h['activation']:.2f}", fontsize=8)

            # Row 2 — concept spatial heatmap overlay (reuse existing attributor)
            ax1 = axes[1][col]
            heatmap = None
            try:
                res = concept_spatial_heatmap(
                    h["image_path"], clip_ft, sae_model, sae_reps[idx], cid,
                    device, concept_match_scores=concept_match_scores,
                    vocab_names=vocab_names, return_only=True,
                )
                if res is not None:
                    _, heatmap = res
            except Exception:
                heatmap = None
            if img_pil is not None:
                w, hgt = img_pil.size
                ax1.imshow(np.array(img_pil), extent=(0, w, hgt, 0))
                if heatmap is not None:
                    ax1.imshow(heatmap, cmap="jet", alpha=0.5,
                               extent=(0, w, hgt, 0), interpolation="bilinear")
            ax1.axis("off")

        fig.suptitle(
            f"Concept {cid}: {cname} — top {n} ft-train images\n"
            "row 1 = image   ·   row 2 = concept heatmap (regions driving the concept)",
            fontsize=10, fontweight="bold",
        )
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"concept_{cid}_{_safe(cname)}.png")
        fig.savefig(out_path, dpi=130, bbox_inches="tight")
        plt.close(fig)
        saved.append(out_path)

    print(f"  Saved top-{top_k} ft-image montages (+ concept heatmaps) for "
          f"{len(saved)} concept(s) → {out_dir}/")
    return saved


def run_maco_parallel(
    candidate_concepts, sae_model, clip_ft, test_ds, sae_dir,
    concept_match_scores, vocab_names, device, args, run_dir, sae_path,
    subfolder=None,
):
    """Dispatch MACO visualization, optionally across worker processes.

    Chooses a platform-appropriate parallelization:
      * CPU / MPS (MACO runs on CPU): N workers, each capped to cpu_count//N
        threads, concepts split round-robin.
      * multi-GPU CUDA: one worker per GPU.
      * single-GPU CUDA: process parallelism can't help — runs serially and
        points to --maco_shard for cross-job (SLURM array) parallelism.

    Falls back to the plain serial call for workers<=1 or a single concept.
    `subfolder` nests output under maco_concepts/<subfolder>/.
    """
    maco_kwargs = dict(
        num_steps=args.maco_num_steps,
        num_crops=args.maco_num_crops,
        reference_samples=args.maco_reference_samples,
        concept_batch=args.maco_concept_batch,
        early_stop_patience=args.maco_early_stop_patience,
        early_stop_delta=args.maco_early_stop_delta,
        shard="0/1",   # cross-run sharding already applied below
        subfolder=subfolder,
    )

    # Apply the cross-run (job-array) shard first, so workers only split the
    # concepts THIS process is responsible for.
    if args.maco_shard and args.maco_shard != "0/1":
        si, sn = (int(v) for v in args.maco_shard.split("/"))
        candidate_concepts = [c for j, c in enumerate(candidate_concepts) if j % sn == si]

    n = len(candidate_concepts)
    workers = max(1, int(getattr(args, "maco_workers", 1)))

    dev_type = torch.device(device).type
    worker_device = "cuda" if dev_type == "cuda" else "cpu"   # mps → cpu for MACO

    def _serial():
        return visualize_candidate_concepts_with_maco(
            candidate_concepts=candidate_concepts, sae_model=sae_model,
            clip_ft=clip_ft, test_ds=test_ds, sae_dir=sae_dir,
            concept_match_scores=concept_match_scores, vocab_names=vocab_names,
            device=device, **maco_kwargs,
        )

    if workers <= 1 or n <= 1:
        return _serial()

    gpus = None
    threads = None
    if dev_type == "cuda":
        ngpu = torch.cuda.device_count()
        if ngpu <= 1:
            print("  MACO: single GPU — process parallelism gives no speedup "
                  "(the GPU is already the bottleneck). Running serially; use "
                  "--maco_shard across a SLURM array for multi-job parallelism.")
            return _serial()
        workers = min(workers, ngpu, n)
        gpus = [i % ngpu for i in range(workers)]
    else:
        workers = min(workers, n)
        total = os.cpu_count() or 4
        threads = max(1, total // workers)

    # Precompute concept→name once so workers don't need the full score matrix.
    concept_names = None
    if vocab_names is not None and concept_match_scores is not None:
        concept_names = {int(cid): vocab_names[concept_match_scores[:, cid].argmax()]
                         for cid in candidate_concepts}

    ref_n = max(1, min(args.maco_reference_samples, len(test_ds.samples)))
    samples = [(s[0], 0, "", 0) for s in test_ds.samples[:ref_n]]
    # Workers reload the SAE from disk rather than pickling it, so they need to
    # know WHICH loader to use -- msae.sae.SAE and RouteSAE take different
    # checkpoint formats and constructor args (RouteSAE's k isn't recoverable
    # from the checkpoint's tensor shapes, so it has to travel with it).
    from routesae import RouteSAE as _RouteSAE
    model_load = dict(
        mode="zeroshot" if args.clip_mode == "zeroshot" else "finetuned",
        clip_model=args.clip_model,
        hf_model_dir=getattr(args, "hf_model_dir", None),   # server offline support
        model_pt=os.path.join(run_dir, "model.pt"),
        sae_path=sae_path,
        sae_kind="routesae" if isinstance(sae_model, _RouteSAE) else "msae",
        routesae_k=getattr(args, "routesae_k", 32),
    )

    # Round-robin split balances early-stopping variance across workers.
    shards = [candidate_concepts[i::workers] for i in range(workers)]
    payloads = [
        dict(
            shard=shards[w],
            device=worker_device,
            gpu=(gpus[w] if gpus else None),
            threads=threads,
            model_load=model_load,
            sae_dir=sae_dir,
            concept_names=concept_names,
            samples=samples,
            maco_kwargs=maco_kwargs,
        )
        for w in range(workers)
    ]

    layout = (f"1 GPU each" if gpus else f"{threads} threads each")
    print(f"  MACO: {workers} parallel workers ({layout}); "
          f"{n} concepts split round-robin (each worker reloads CLIP+SAE).")

    import torch.multiprocessing as _mp
    ctx = _mp.get_context("spawn")
    saved = []
    with ctx.Pool(processes=workers) as pool:
        for res in pool.map(_maco_parallel_worker, payloads):
            saved.extend(res or [])
    return saved


def _weighted_concept_matrix(concepts, sae_model, w=None):
    """
    Build W, the (K_f, d) matrix whose rows are the (weighted) SAE decoder
    directions for `concepts` -- shared setup for both projection-matrix
    builders below (qr_decompose_concepts, pinv_projection_matrix).

    Row j is  w_j * f_j,  where f_j is concept j's decoder direction. If w
    is None, uniform weights 1/K_f are assigned so all concepts contribute
    equally.

    RouteSAE's decoder lives in its own normalized (per-position centered/
    scaled) input space -- no projection through CLIP's output head needed
    here (unlike select_dialguided_concepts, which compares against text
    embeddings and so DOES need that projection). hook_routesae_projection
    normalizes the residual stream the same way before projecting, so the
    spaces match; MSAE's decoder needs no such step since it already lives
    directly in the (unnormalized) CLIP embedding space projection_ablation_hook
    operates on.

    Parameters
    ----------
    concepts  : list[int]          filtered concept IDs  (K_f entries)
    sae_model : SAE | RouteSAE     model whose decoder rows are concept directions
    w         : Tensor (K,) | None weight vector from denois_candidate_concepts
                                   (full-length, may include zeros for dropped concepts);
                                   pass None to use uniform weights

    Returns
    -------
    W : Tensor (K_f, d) | None   None if concepts is empty
    """
    if not concepts:
        return None

    from routesae import RouteSAE
    if isinstance(sae_model, RouteSAE):
        dec = sae_model.sae.decoder.weight.T                     # (latent_size, hidden_size)
    else:
        _ae = sae_model.model
        dec = _ae.decoder if _ae.decoder is not None else _ae.encoder.t()
    f_filtered = dec[concepts].detach().float()                  # (K_f, d)
    K_f        = len(concepts)

    if w is None:
        w_filtered = torch.full((K_f, 1), 1.0 / K_f,
                                dtype=torch.float32, device=f_filtered.device)
    else:
        w_filtered = w[w > 0].float().unsqueeze(1)               # (K_f, 1)

    return w_filtered * f_filtered                                # (K_f, d)


def qr_decompose_concepts(concepts, sae_model, w=None):
    """
    QR-decompose the (weighted) concept matrix's transpose, so Q has
    orthonormal columns spanning the same subspace, ready for projection
    ablation: feat - (feat @ Q) @ Q.T  (i.e. feat @ (Q @ Q.T)).

    See _weighted_concept_matrix for the concepts/sae_model/w parameters.

    Returns
    -------
    Q : Tensor (d, K_f)   orthonormal basis spanning the weighted concept subspace
    R : Tensor (K_f, K_f) upper-triangular factor;  V_m.T = Q @ R
    """
    V_m = _weighted_concept_matrix(concepts, sae_model, w)
    if V_m is None:
        return torch.empty(0), torch.empty(0)

    _qr_dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    Q, R    = torch.linalg.qr(V_m.T.to(_qr_dev))               # Q:(d,K_f)  R:(K_f,K_f)
    Q, R    = Q.to(V_m.device), R.to(V_m.device)

    return Q, R


def pinv_projection_matrix(concepts, sae_model, w=None):
    """
    Build the orthogonal projection matrix onto span(W) directly, without
    orthonormalizing first:

        P = W.T @ pinv(W @ W.T) @ W          W = (weighted) concept decoder
                                              directions, (K_f, d)

    This is the standard closed-form projector onto the row space of W
    (any full-rank basis of a subspace gives the same P as an orthonormal
    one -- projection onto a subspace doesn't depend on which basis you
    describe it with). Uses torch.linalg.pinv (SVD-based) rather than a
    literal matrix inverse: decoder directions for distinct concept IDs can
    still be near-duplicate/near-parallel (e.g. two latents both naming to
    "shoulder"), which makes W @ W.T near-singular -- pinv degrades
    gracefully there where .inverse() would blow up or error outright.

    See _weighted_concept_matrix for the concepts/sae_model/w parameters.

    Returns
    -------
    P : Tensor (d, d)   symmetric idempotent projection matrix; feat @ P is
                        feat's component inside span(W). Empty if concepts is empty.
    """
    W = _weighted_concept_matrix(concepts, sae_model, w)
    if W is None:
        return torch.empty(0)

    orig_device = W.device
    _dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    W = W.to(_dev)
    gram = W @ W.T                                               # (K_f, K_f)
    P = W.T @ torch.linalg.pinv(gram) @ W                        # (d, d)

    return P.to(orig_device)


def build_projection_matrix(concepts, sae_model, method="qr", w=None):
    """Dispatch to qr_decompose_concepts or pinv_projection_matrix per
    --projection_method, returning a (d, d) projection matrix P ready for
    projection_ablation_hook (MSAE) or RouteSAE's projection path -- both
    consume P the same way regardless of which method built it.
    """
    if method == "pinv":
        return pinv_projection_matrix(concepts, sae_model, w=w)
    Q, _R = qr_decompose_concepts(concepts, sae_model, w=w)
    if Q.numel() == 0:
        return Q
    return Q @ Q.T


# ── [PRISM] Zero-shot evaluation with orthogonal projection ────────────────────
# Adapted from https://github.com/MahdiyarMM/PRISM/blob/main/utils.py
#   orth_transformation_calculation  →  prism_orth_projection
#   classify_images                  →  prism_classify
#   accuracy_by_subgroup             →  prism_accuracy_by_subgroup
# Enabled by --prism_eval flag; all hyperparameters are configurable via args.
# Group names reuse GROUP_NAMES defined above — no duplication.


def prism_encode_text(
    clip_model,
    class_names: list,   #Capitalised class names, e.g. ["Landbird", "Waterbird"]
    device,
    templates: tuple = ("a photo of a {}.", "a picture of a {}."),   ### use photo and picture together and average over them
) -> torch.Tensor:
    """[PRISM] Encode class names with multiple templates and average per class.

    PRISM main.py averages embeddings over templates per class before
    normalising, giving a more robust text anchor than a single prompt.

    Returns tensor (n_classes, D) on CPU.
    """
    import clip as _clip
    prompts = [t.format(c) for c in class_names for t in templates]
    tokens  = _clip.tokenize(prompts).to(device)
    with torch.no_grad():
        emb = clip_model.encode_text(tokens).float()
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    emb = emb.view(len(class_names), len(templates), -1).mean(dim=1)
    emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    return emb.cpu()


def prism_orth_projection(clip_model, spurious_phrases: list, device) -> torch.Tensor:
    """[PRISM] Build orthogonal projection matrix that removes the spurious subspace.

    Adapted from orth_transformation_calculation() in PRISM/utils.py.
    Encodes spurious phrases → direction matrix V (D, k), then computes:
        P = I  -  V (VᵀV)⁻¹ Vᵀ
    Applying P to an image embedding removes its component along the spurious directions.
    Returns P of shape (D, D) on CPU.
    """
    import clip as _clip
    tokens = _clip.tokenize(spurious_phrases).to(device)
    with torch.no_grad():
        V = clip_model.encode_text(tokens).float()
    V = V / V.norm(dim=-1, keepdim=True).clamp(min=1e-8)
    V = V.T                               # (D, k)
    VtV_inv = torch.inverse(V.T @ V)      # (k, k)^{-1}
    P = torch.eye(V.shape[0], device=device) - V @ VtV_inv @ V.T   # (D, D)
    return P.cpu()


def prism_classify(
    clip_model,
    text_embeddings: torch.Tensor,
    dataset,
    device: str,
    batch_size: int = 64,
    projection: "torch.Tensor | None" = None,
    description: str = "PRISM classify",
) -> dict:
    """[PRISM] Zero-shot classification by cosine similarity with optional projection.

    Adapted from classify_images() in PRISM/utils.py.
    dataset.clip_preprocess must be set before calling this function.
    Returns dict with predictions, true_labels, places, group_ids, accuracy.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        num_workers=0, pin_memory=False)
    te = text_embeddings.float().to(device)
    P  = projection.float().to(device) if projection is not None else None
    all_preds, all_labels, all_gids = [], [], []
    with torch.no_grad():
        for imgs, labels, _, gids in tqdm(loader, desc=f"  {description}"):
            imgs = imgs.to(device)
            emb  = clip_model.encode_image(imgs).float()
            emb  = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            if P is not None:
                emb = emb @ P
                emb = emb / emb.norm(dim=-1, keepdim=True).clamp(min=1e-8)
            preds = (emb @ te.T).argmax(dim=-1)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu() if isinstance(labels, torch.Tensor)
                               else torch.tensor(labels))
            all_gids.append(gids.cpu() if isinstance(gids, torch.Tensor)
                            else torch.tensor(gids))
    preds_t  = torch.cat(all_preds)
    labels_t = torch.cat(all_labels)
    gids_t   = torch.cat(all_gids)
    correct  = (preds_t == labels_t).sum().item()
    total    = len(labels_t)
    print(f"  {description}: {correct}/{total} = {correct / total * 100:.2f}%")
    return {"predictions": preds_t, "true_labels": labels_t,
            "places": gids_t % 2, "group_ids": gids_t,
            "accuracy": correct / total * 100}


def prism_accuracy_by_subgroup(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    spurious: torch.Tensor,
) -> dict:
    """[PRISM] Per-subgroup accuracy for all (y, spurious background) combinations.

    Adapted from accuracy_by_subgroup() in PRISM/utils.py.
    Reuses GROUP_NAMES dict defined in this module (no duplication).
    Reports adjusted average accuracy (equal weight per group) and worst-group accuracy.
    """
    results, worst_acc, any_valid = {}, 1.0, False
    print(f"\n  {'Subgroup':<34} {'Acc%':>7}  {'Correct':>8}  {'Total':>7}")
    print("  " + "─" * 62)
    for y_val in [0, 1]:
        for s_val in [0, 1]:
            mask = (labels == y_val) & (spurious == s_val)
            tot  = int(mask.sum())
            if tot == 0:
                acc, corr = float("nan"), 0
            else:
                corr = int((predictions[mask] == labels[mask]).sum())
                acc  = corr / tot
                worst_acc = min(worst_acc, acc)
                any_valid = True
            gname = GROUP_NAMES[y_val * 2 + s_val]
            key   = f"y={y_val}_s={s_val}"
            acc_s = f"{acc * 100:.1f}%" if tot else "    N/A"
            results[key] = {"acc": acc * 100 if tot else float("nan"),
                            "correct": corr, "total": tot, "group": gname}
            print(f"  {gname:<34} {acc_s:>7}  {corr:>8}  {tot:>7}")
    wga     = worst_acc * 100 if any_valid else float("nan")
    valid_a = [v["acc"] for v in results.values()
               if isinstance(v, dict) and v["total"] > 0 and not np.isnan(v["acc"])]
    adj_avg = sum(valid_a) / len(valid_a) if valid_a else float("nan")
    results["worst_group_acc"] = wga
    results["adj_acc_avg"]     = adj_avg
    print(f"\n  Adjusted avg accuracy: {adj_avg:.2f}%  (equal weight per group)")
    print(f"  Worst-group accuracy : {wga:.2f}%")
    return results


def _load_spurious_attrs_from_files(path: str, attr_type: str) -> dict:
    """Load spurious attribute word lists from text files inside sae_dir/attribute_words/."""
    
    def _read(fname):
        with open(os.path.join(path, fname), "r") as f:
            return [line.strip() for line in f if line.strip()]
    
    if attr_type == "spurious":
        return {
            "landbird":  _read("ground_backgrounds.txt"),
            "waterbird": _read("water_backgrounds.txt"),
        }
    elif attr_type == "good":
        return {
            "landbird":  _read("land_birds.txt"),
            "waterbird": _read("water_birds.txt"),
        }

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    global _hf_model_dir
    args = parse_args()
    if args.hf_model_dir:
        _hf_model_dir = args.hf_model_dir
    if args.output_dir is None:
        args.output_dir = os.path.join(".", "results", args.dataset)
    concept_extractor_name = args.clip_mode
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

    # ── Load CLIP model (zero-shot or fine-tuned) ─────────────────────────────
    device = (
        "cuda" if torch.cuda.is_available()
        else "mps"  if torch.backends.mps.is_available()
        else "cpu"
    )
    if args.clip_mode == "zeroshot":
        clip_ft = CLIPZeroShot(model_name=args.clip_model, device=device)
        print(f"\nZero-shot CLIP loaded  ({args.clip_model}, device={device})")
    else:
        clip_ft = CLIPZeroShot.load_model(os.path.join(run_dir, "model.pt"), device=device)
        print(f"\nFine-tuned CLIP loaded  (device={device})")
    clip_ft.model.eval()

    # ── [PRISM] Orthogonal projection debiasing ───────────────────────────────
    if args.prism_eval:
        from prism_utils import (
            orth_transforamtion_calculation,
            classify_images as _prism_classify_images,
            accuracy_by_subgroup as _prism_accuracy_by_subgroup,
        )
        import clip as _clip_mod

        test_ds.clip_preprocess = clip_ft.preprocess
        _prism_cls  = args.prism_class_names
        _prism_tmpl = list(args.prism_templates)
        _prism_spur =  args.prism_spurious_words # land and water only
    #_load_spurious_attrs_from_files(sae_dir) 
        # Minimal args namespace matching what prism_utils functions expect
        _prism_args = argparse.Namespace(device=device, batch_size=args.batch_size)

        # Wrap ManifestDataset to return PRISM-compatible 3-tuples:
        # (image, label, metadata) where metadata[:,0] = background (0=land, 1=water)
        class _PRISMCompatDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                img, label, bg_name, gid = self.ds[idx]
                bg_int = 1 if bg_name == "water" else 0
                metadata = torch.tensor([bg_int, label], dtype=torch.long)
                return img, label, metadata

        _prism_loader = torch.utils.data.DataLoader(
            _PRISMCompatDataset(test_ds),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

        # Text encoding — exactly as PRISM main.py:
        # tokenise all (class × template) combos, encode, normalise, average per class, re-normalise
        # text_prompts = [t.format(c) for c in _prism_cls for t in _prism_tmpl]
        # text_tokens = _clip_mod.tokenize(text_prompts).to(device)
        # with torch.no_grad():
        #     text_embeddings = clip_ft.model.encode_text(text_tokens)
        #     text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        # text_embeddings = text_embeddings.view(
        #     len(_prism_cls), -1, text_embeddings.shape[-1]
        # ).mean(dim=1)
        # text_embeddings /= text_embeddings.norm(dim=-1, keepdim=True)
        # text_embeddings = text_embeddings.to(torch.float32)
        _prism_cls  = ["Landbird", "Waterbird"]
        _prism_tmpl = ["a photo of a {}.", "a picture of a {}."]
        text_embeddings = prism_encode_text(
            clip_ft.model, _prism_cls, device, templates=tuple(_prism_tmpl)
        ).to(device)

        print("\n" + "─" * 65)
        print("  [PRISM] ZERO-SHOT — NO PROJECTION (baseline)")
        print(f"  Model    : {args.clip_model}")
        print(f"  Classes  : {_prism_cls}  |  Templates: {_prism_tmpl}")
        print("─" * 65)
        _, _, [all_y, all_preds, all_metadata] = _prism_classify_images(
            _prism_args, clip_ft.model, text_embeddings, _prism_loader,
            P=None, description="PRISM ZS",
        )
        _prism_accuracy_by_subgroup(
            list(all_preds.cpu().numpy()),
            list(all_y.cpu().numpy()),
            [x[0] for x in list(all_metadata.cpu().numpy())],
        )

        print("\n" + "─" * 65)
        print("  [PRISM] ZERO-SHOT + ORTHOGONAL PROJECTION")
        print(f"  Spurious words: {_prism_spur}")
        print("─" * 65)
        _P = orth_transforamtion_calculation(_prism_args, clip_ft.model, _prism_spur)
        _, _, [all_y, all_preds, all_metadata] = _prism_classify_images(
            _prism_args, clip_ft.model, text_embeddings, _prism_loader,
            P=_P, description="PRISM ZS+Proj",
        )
        _prism_accuracy_by_subgroup(
            list(all_preds.cpu().numpy()),
            list(all_y.cpu().numpy()),
            [x[0] for x in list(all_metadata.cpu().numpy())],
        )

    # ── SAE representation extraction (optional) ──────────────────────────────
    from msae.sae import SAE

    report_suffix = "__denoised" if args.denoise_concepts else ""
    if args.model == "RouteSAE":
        # RouteSAE is architecturally unrelated to msae/train.py's SAE family
        # (per-layer/per-patch residual-stream SAE, not a single-embedding one)
        # -- no auto-discovery, no n_inputs/activation-string filename matching,
        # just load the exact checkpoint given.
        from routesae_adapter import load_routesae_for_clip, extract_routesae_representations

        if not args.sae_path or not os.path.isfile(args.sae_path):
            print(f"\n--model RouteSAE requires --sae_path pointing directly at a checkpoint "
                  f"file (no auto-discovery). Got: {args.sae_path!r}")
            return
        # RouteSAE's hook-based extraction (routesae.py) explicitly upcasts
        # intermediate activations to fp32, but CLIP models loaded for non-CPU
        # devices are often cast to fp16 -- mismatched dtype crashes at the
        # first matmul against a still-fp16 weight (e.g. visual.proj). RouteSAE
        # itself is fp32, so force the CLIP model to match rather than chase
        # every individual op that trips over the mismatch.
        clip_ft.model = clip_ft.model.float()
        print(f"\nLoading RouteSAE from: {args.sae_path}  (k={args.routesae_k})")
        sae_model = load_routesae_for_clip(args.sae_path, device=device, k=args.routesae_k)
        # denois_candidate_concepts filters by cosine similarity to the mean
        # SAE *decoder* direction -- a single global decoder matrix, which
        # RouteSAE (per-layer/per-patch, no single decoder) doesn't have.
        # --denoise_concepts has no CLI way to disable (store_true +
        # default=True), so force it off here instead.
        if args.denoise_concepts:
            print("  Note: --denoise_concepts has no effect for --model RouteSAE "
                  "(no single decoder matrix to compute directions from) -- disabling it.")
            args.denoise_concepts = False
        # sae_path/embedding_path are only set in the MSAE branch below but read
        # unconditionally further down (vocab_path lookup, MACO's sae_path arg) --
        # set them here too, same formula as the MSAE branch.
        sae_path = args.sae_path
        embedding_path = os.path.join(os.path.dirname(run_dir), "embeddings")
        # Self-descriptive: routesae checkpoints already encode K/latent_size/source
        # in their own filename (e.g. routesae_K32_clip_ft_BIASED_400_..._16384).
        sae_tag = os.path.splitext(os.path.basename(args.sae_path))[0]

        clip_tag = "zs" if args.clip_mode == "zeroshot" else "ft"
        base_sae_dir = os.path.join(run_dir, f"{clip_tag}_{sae_tag}")
        sae_dir = os.path.join(base_sae_dir, f"knn{args.knn_k}_prev{args.prevalence_threshold}")
        os.makedirs(base_sae_dir, exist_ok=True)
        os.makedirs(sae_dir, exist_ok=True)
        print(f"  Representations folder: {base_sae_dir}")
        print(f"  Analysis output folder: {sae_dir}")

        print("\nExtracting RouteSAE representations — ft_train...")
        ft_results_path = os.path.join(base_sae_dir, "representations_ft_train")
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
            ft_results = extract_routesae_representations(
                clip_ft.model, sae_model, ft_ds, device, clip_ft.preprocess)
            save_representations(ft_results, "ft_train", base_sae_dir)

        print("\nExtracting RouteSAE representations — test...")
        te_results_path = os.path.join(base_sae_dir, "representations_test")
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
            te_results = extract_routesae_representations(
                clip_ft.model, sae_model, test_ds, device, clip_ft.preprocess)
            save_representations(te_results, "test", base_sae_dir)

    else:
        # resolve feature file and embed dim from previously saved .npy
        embedding_path = os.path.join(os.path.dirname(run_dir), "embeddings")
        model_tag = args.clip_model.replace("/", "~")
        if args.clip_mode == "zeroshot":
            feat_npy_files = glob.glob(os.path.join(embedding_path, f"{args.dataset}_{model_tag}_zs*_train_*.npy"))
            feat_label = "zs"
        else:
            feat_npy_files = glob.glob(os.path.join(embedding_path, f"{args.dataset}_{model_tag}_ft*_fttrain_*.npy"))
            feat_label = "ft"
        if not feat_npy_files:
            print(f"\nNo {feat_label} feature file found in {embedding_path}. Run feature extraction first.")
            return
        feat_path = max(feat_npy_files, key=os.path.getmtime)
        d = int(os.path.basename(feat_path).replace(".npy", "").split("_")[-1])
        print(f"\n{feat_label.upper()} feature file : {feat_path}  (embed_dim={d})")

        # ── Build activation string exactly as train.py does ────────────────────────
        _act = args.activation
        if args.model == "ReLUSAE" and "_" in _act:
            _act_base, _sparse_w = _act.split("_", 1)
            activation_str = f"{_act_base}_{str(float(f'0.{_sparse_w}')).split('.')[1]}"
        elif args.model in ["MSAE_UW", "MSAE_RW"]:
            if args.activation is None:
                model = args.model.replace("MSAE_", "")
                activation_str = "TopK_64" + "_" + model
            else:
                activation_str = args.activation
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
            # auto-locate SAE weights that match the clip_mode
            model_tag = args.clip_model.replace("/", "~")
            if args.clip_mode == "zeroshot":
                sae_weights_pattern = os.path.join(
                    "results",
                    f"{args.dataset}_{model_tag}_zs_train_image_*",
                    "sae_weights", "*.pth",
                )
            else:
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

        # Representations depend only on clip_mode/SAE — shared across knn_k/prevalence runs.
        clip_tag = "zs" if args.clip_mode == "zeroshot" else "ft"
        base_sae_dir = os.path.join(run_dir, f"{clip_tag}_{sae_tag}")
        # Analysis outputs (plots, concept pools, reports) are parameter-specific.
        sae_dir = os.path.join(base_sae_dir, f"knn{args.knn_k}_prev{args.prevalence_threshold}")
        os.makedirs(base_sae_dir, exist_ok=True)
        os.makedirs(sae_dir, exist_ok=True)
        print(f"  Representations folder: {base_sae_dir}")
        print(f"  Analysis output folder: {sae_dir}")

        print("\nExtracting SAE representations — ft_train...")
        ft_results_path = os.path.join(base_sae_dir, "representations_ft_train")
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
            save_representations(ft_results, "ft_train", base_sae_dir)

        print("\nExtracting SAE representations — test...")
        te_results_path = os.path.join(base_sae_dir, "representations_test")
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
            save_representations(te_results, "test", base_sae_dir)

    # ── Zero-shot accuracy report (per-group + adjusted average) ─────────────
    if args.clip_mode == "zeroshot":
        TRAIN_SIZES = {0: 3498, 1: 184, 2: 56, 3: 1057}
        _total_train = sum(TRAIN_SIZES.values())

        _zs = clip_ft.run(dataset=test_ds, prompt_mode="shape", dataset_name="waterbirds")
        _preds  = np.array(_zs["predictions_shape"])
        _labels = np.array(_zs["true_labels"])
        _gids   = np.array([s[3] for s in test_ds.samples])

        overall_acc = (_preds == _labels).mean() * 100

        print("\n" + "─" * 65)
        print("  [ZERO-SHOT] Test set accuracy")
        print("─" * 65)
        print(f"  {'Group':<42}  {'Acc':>7}  {'Correct':>8}  {'Total':>6}")
        print(f"  {'─'*42}  {'─'*7}  {'─'*8}  {'─'*6}")

        per_group_acc = {}
        for gid in sorted(set(_gids.tolist())):
            mask    = _gids == gid
            correct = int((_preds[mask] == _labels[mask]).sum())
            total   = int(mask.sum())
            acc     = correct / total * 100
            per_group_acc[gid] = acc
            print(f"  Group {gid} ({GROUP_NAMES.get(gid, str(gid)):<35})  {acc:>6.2f}%  {correct:>8}  {total:>6}")

        adj_acc = sum(
            per_group_acc[gid] * TRAIN_SIZES[gid] / _total_train
            for gid in TRAIN_SIZES if gid in per_group_acc
        )
        wga = min(per_group_acc.values())

        print(f"  {'─'*65}")
        print(f"  Overall accuracy                          :  {overall_acc:>6.2f}%")
        print(f"  Adjusted avg accuracy (train-weighted)    :  {adj_acc:>6.2f}%")
        print(f"  Worst-group accuracy                      :  {wga:>6.2f}%")
        print("─" * 65)

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
    
    # Select the concept_match scores computed for THIS SAE. sae_naming.py names
    # its output Concept_Interpreter_<sae_weights>_<vocab>.npy, so we match on the
    # SAE weights basename rather than blindly taking the newest file (which could
    # belong to a different SAE and silently mis-shape the results). When several
    # vocabularies were named for this SAE, --concept_match_vocab picks one.
    sae_basename = os.path.splitext(os.path.basename(sae_path))[0]
    if os.path.isdir(vocab_path):
        candidates = sorted(glob.glob(os.path.join(vocab_path, "*.npy")))
        matches = [p for p in candidates if sae_basename in os.path.basename(p)]
        if args.concept_match_vocab:
            matches = [p for p in matches
                       if args.concept_match_vocab in os.path.basename(p)]
        if not matches:
            raise FileNotFoundError(
                f"No concept_match .npy for SAE '{sae_basename}'"
                + (f" + vocab '{args.concept_match_vocab}'" if args.concept_match_vocab else "")
                + f" in {vocab_path}.\nRun msae/sae_naming.py for this SAE/vocab first."
            )
        vocab_path_to_load = max(matches, key=os.path.getmtime)
    elif os.path.isfile(vocab_path):
        vocab_path_to_load = vocab_path
    else:
        raise FileNotFoundError(
            f"Concept_match path not found: {vocab_path}\n"
            f"(neither a directory of Concept_Interpreter_*.npy files, nor a single "
            f".npy file). This usually means concept naming (msae/sae_naming.py or "
            f"routesae_naming.py) was never run for this SAE against --run_dir "
            f"{run_dir!r} -- check --run_dir points at the run this checkpoint's "
            f"naming/concept_match artifacts actually belong to."
        )

    concept_match_scores = np.load(vocab_path_to_load)
    print(f"Loaded concept_match : {os.path.basename(vocab_path_to_load)}")
    print("concept_match_scores.shape:", concept_match_scores.shape)

    # Load the vocab names ROW-ALIGNED to the selected scores. sae_naming embeds
    # <vocab>_<dim>.npy whose aligned names live in the companion <vocab>.txt
    # (dim suffix dropped) in the embeddings dir. Deriving the names from the
    # chosen file — instead of always reading clip_disect_20k.txt — keeps names
    # correct across vocabularies (disect, waterbirds_domain, …).
    vocab_names = None
    prefix = f"Concept_Interpreter_{sae_basename}_"
    base = os.path.splitext(os.path.basename(vocab_path_to_load))[0]
    if base.startswith(prefix):
        vocab_tag = re.sub(r"_\d+$", "", base[len(prefix):])   # drop trailing _<dim>
        names_txt = os.path.join(embedding_path, vocab_tag + ".txt")
        if os.path.isfile(names_txt):
            with open(names_txt) as f:
                vocab_names = [line.strip() for line in f if line.strip()]
    if vocab_names is None:
        print("  [warn] aligned names file not found; falling back to clip_disect_20k.txt")
        with open('msae/vocab/clip_disect_20k.txt', 'r') as f:
            vocab_names = [line.strip() for line in f if line.strip()]

    if len(vocab_names) != concept_match_scores.shape[0]:
        raise ValueError(
            f"vocab_names ({len(vocab_names)}) does not match concept_match rows "
            f"({concept_match_scores.shape[0]}). The names file is not aligned to "
            f"the selected scores: {vocab_path_to_load}"
        )
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
    # candidate_concepts, concept_source_group, concept_mis_prevalence = find_spurious_concepts(
    #     te_results=te_results,
    #     ft_results=ft_results,
    #     test_ds=test_ds,
    #     ft_ds=ft_ds,
    #     concept_match_scores=concept_match_scores,
    #     vocab_names=vocab_names,
    #     sae_dir=sae_dir,
    #     clip_ft=clip_ft,
    # )
### Experiment set 5: Ablation of candidate spurious concepts — zero out the corresponding SAE latents, reconstruct CLIP features, re-classify, and report accuracy deltas for each group along with example corrected/worsened images.
    # candidate_concepts, concept_source_group, concept_mis_prevalence = find_spurious_concepts_binary(
    #     te_results=te_results,
    #     ft_results=ft_results,
    #     test_ds=test_ds,
    #     ft_ds=ft_ds,
    #     concept_match_scores=concept_match_scores,
    #     vocab_names=vocab_names,
    #     sae_dir=sae_dir,
    #     clip_ft=clip_ft,
    # )

    

    # parser.add_argument("--concept_finding_method", type=str, default="labelfree",
    #                     choices=["labelfree", "labelguided"],
    #                     help="Method for finding candidate concepts: "
    #                          "'labelfree' uses the SAE to find high-magnitude concepts; "
    #                          "'labelguided' uses the label to find concepts correlated with it.")
    # parser.add_argument("--denoise_concepts", action="store_true", default=True,
    #                     help="If set, denoise candidate concepts by removing those "
    #                          "who are far from the mean activation.")
    # parser.add_argument("--editing_method", type=str, default="deactivation",
    #                     choices=["deactivation", "projection"],
    #                     help="Method for editing concepts: "
    #                          "'deactivation' zeroes out the concept; "
    #                          "'projection' projects the representation onto the orthogonal complement.")
        
    if args.concept_finding_method == "labelfree":

        concept_extractor_name = concept_extractor_name + "_labelfree"
        M, Y_hat, m_centroid, m_knn = candidate_selection(te_results, clip_ft, k=args.knn_k)
        candidate_image_paths = [te_results["image_paths"][i] for i in M.nonzero(as_tuple=True)[0].tolist()]

        Save_candidate_report(
            M=M, Y_hat=Y_hat, M_centroid=m_centroid, M_knn=m_knn,
            te_results=te_results, test_ds=test_ds,
            save_path=os.path.join(sae_dir, "candidate_report.csv"),
        )

        # records = extract_active_concepts(
        #     image_paths=candidate_image_paths,
        #     clip_ft=clip_ft,
        #     sae_model=sae_model,
        #     vocab_names=vocab_names,
        #     concept_match_scores=concept_match_scores,
        # )
        

        concept_pool, per_image_pool = Generate_Concept_Pool(
            te_results=te_results,
            test_ds=test_ds,
            sae_model=sae_model,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            sae_dir=base_sae_dir,
            clip_ft=clip_ft,
            batch_size=args.concept_pool_batch_size,
        )

        # ── Select concepts persistently influential in candidate images following labelfree ──────────
        candidate_concepts, concept_counts = select_prevalent_concepts(
            M=M,
            per_image_pool=per_image_pool,
            concept_pool=concept_pool,
            prevalence_threshold=args.prevalence_threshold,
            top_k=args.top_k_concepts,
            n_cpu_workers=args.n_cpu_workers,
        )

    elif args.concept_finding_method == "dialguided":
        concept_extractor_name = concept_extractor_name + "_dialguided"
        candidate_concepts = select_dialguided_concepts(
            te_results=te_results,
            ft_results=ft_results,
            test_ds=test_ds,
            ft_ds=ft_ds,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            sae_dir=sae_dir,
            clip_ft=clip_ft,
            sae_model=sae_model, device=device,
            sp_arribute_dir = "results/" + args.dataset + "/attribute_words",
            top_n_concepts=args.top_k_concepts
            
        )
        
    elif args.concept_finding_method == "highmag":
        concept_extractor_name = concept_extractor_name + "_highmag"
        candidate_concepts, concept_source_group, concept_mis_prevalence = find_spurious_concepts_highmag(
            te_results=te_results,
            ft_results=ft_results,
            test_ds=test_ds,
            ft_ds=ft_ds,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            sae_dir=sae_dir,
            clip_ft=clip_ft,
            top_n_concepts=args.top_k_concepts,
        )

    elif args.concept_finding_method == "labelguided":
        concept_extractor_name = concept_extractor_name + "_labelguided"
        candidate_concepts, concept_source_group, concept_mis_prevalence = find_and_show_spurious_concepts_binary(
            te_results=te_results,
            ft_results=ft_results,
            test_ds=test_ds,
            ft_ds=ft_ds,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            sae_dir=sae_dir,
            clip_ft=clip_ft,
            sae_model=sae_model, device=device,
        )
    elif args.concept_finding_method == "none":
        # No concept-finding at all -- ablation runs with an empty concept
        # list below, which for both make_sae_ablation_hook/routesae_embeds
        # means "encode then decode, zero nothing" -- i.e. plain SAE
        # reconstruction, with no concept deliberately removed.
        concept_extractor_name = concept_extractor_name + "_none"
        candidate_concepts = []
        concept_source_group = None
        concept_mis_prevalence = None
    elif args.concept_finding_method == "prism_baseline":
        print("No concept extraction, Running PRISM baseline evaluation...")
        if args.editing_method != "prism_baseline":
            raise ValueError("PRISM baseline does not support concept related editing.")

        import matplotlib.pyplot as plt
        from PIL import Image as _PIL
        from prism_utils import (
            orth_transforamtion_calculation,
            classify_images as _prism_classify_images,
            accuracy_by_subgroup as _prism_accuracy_by_subgroup,
        )

        METHOD_TAG  = "prism_baseline"
        HOOK_TAG    = "orth_proj"
        CLASS_NAMES = ["landbird", "waterbird"]
        sp_arribute_dir = "results/" + args.dataset + "/attribute_words"
        out_dir = os.path.join(sae_dir, METHOD_TAG, HOOK_TAG)
        os.makedirs(out_dir, exist_ok=True)

        _prism_args = argparse.Namespace(device=device, batch_size=args.batch_size)
        _prism_cls  = ["Landbird", "Waterbird"]
        _prism_tmpl = ["a photo of a {}.", "a picture of a {}."]
        SPURIOUS_ATTRS = _load_spurious_attrs_from_files(sp_arribute_dir,attr_type="spurious")
        # _prism_spur = args.prism_spurious_words

        _prism_spur = SPURIOUS_ATTRS["landbird"] + SPURIOUS_ATTRS["waterbird"]

        class _PRISMCompatDataset(torch.utils.data.Dataset):
            def __init__(self, ds):
                self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, idx):
                img, label, bg_name, gid = self.ds[idx]
                bg_int = 1 if bg_name == "water" else 0
                metadata = torch.tensor([bg_int, label], dtype=torch.long)
                return img, label, metadata

        test_ds.clip_preprocess = clip_ft.preprocess
        _prism_loader = torch.utils.data.DataLoader(
            _PRISMCompatDataset(test_ds),
            batch_size=args.batch_size, shuffle=False, num_workers=0,
        )

        text_embeddings = prism_encode_text(
            clip_ft.model, _prism_cls, device, templates=tuple(_prism_tmpl)
        ).to(device)

        group_ids   = np.array([s[3] for s in test_ds.samples])
        true_labels = np.array([s[1] for s in test_ds.samples])

        # ── Baseline: no projection ───────────────────────────────────────────
        print("\n" + "─" * 65)
        print("  [PRISM] ZERO-SHOT — NO PROJECTION (baseline)")
        print("─" * 65)
        _, _, [all_y, orig_preds_t, all_metadata] = _prism_classify_images(
            _prism_args, clip_ft.model, text_embeddings, _prism_loader,
            P=None, description="PRISM ZS",
        )
        orig_preds = orig_preds_t.cpu().numpy()
        _prism_accuracy_by_subgroup(
            list(orig_preds_t.cpu().numpy()),
            list(all_y.cpu().numpy()),
            [x[0] for x in list(all_metadata.cpu().numpy())],
        )

        # ── With orthogonal projection ────────────────────────────────────────
        print("\n" + "─" * 65)
        print("  [PRISM] ZERO-SHOT + ORTHOGONAL PROJECTION")
        print(f"  Spurious words: {_prism_spur}")
        print("─" * 65)
        _P = orth_transforamtion_calculation(_prism_args, clip_ft.model, _prism_spur)
        _, _, [all_y, proj_preds_t, all_metadata] = _prism_classify_images(
            _prism_args, clip_ft.model, text_embeddings, _prism_loader,
            P=_P, description="PRISM ZS+Proj",
        )
        proj_preds = proj_preds_t.cpu().numpy()
        _prism_accuracy_by_subgroup(
            list(proj_preds_t.cpu().numpy()),
            list(all_y.cpu().numpy()),
            [x[0] for x in list(all_metadata.cpu().numpy())],
        )

        # ── Report (same format as ablate_spurious_concepts) ─────────────────
        report_lines = [
            "PRISM Orthogonal Projection Evaluation",
            "=" * 80,
            f"Spurious words projected out: {_prism_spur}",
            f"Class names : {_prism_cls}",
            f"Templates   : {_prism_tmpl}",
            "-" * 80,
            f"{'Group':<42}  {'No Proj':>9}  {'Orth Proj':>11}  {'Δ Acc':>7}",
            "-" * 74,
        ]
        for gid in sorted(set(group_ids.tolist())):
            mask     = group_ids == gid
            orig_acc = (orig_preds[mask] == true_labels[mask]).mean()
            proj_acc = (proj_preds[mask] == true_labels[mask]).mean()
            report_lines.append(
                f"Group {gid} ({GROUP_NAMES.get(gid, str(gid)):<35})  "
                f"{orig_acc*100:>8.1f}%  {proj_acc*100:>10.1f}%  "
                f"{(proj_acc - orig_acc)*100:>+6.1f}%"
            )
        overall_orig = (orig_preds == true_labels).mean()
        overall_proj = (proj_preds == true_labels).mean()
        report_lines += [
            "-" * 74,
            f"{'Overall':<42}  {overall_orig*100:>8.1f}%  {overall_proj*100:>10.1f}%  "
            f"{(overall_proj - overall_orig)*100:>+6.1f}%",
        ]
        report_text = "\n".join(report_lines) + "\n"
        report_path = os.path.join(out_dir, f"ablation_report_{METHOD_TAG}__{HOOK_TAG}.txt")
        with open(report_path, "w") as f:
            f.write(report_text)
        print(report_text)
        print(f"Report saved to {report_path}")

        # ── Corrected images (groups 1, 2): wrong → correct after projection ─
        for gid in [1, 2]:
            g_indices = np.where(group_ids == gid)[0]
            if len(g_indices) == 0:
                continue
            was_wrong   = orig_preds[g_indices] != true_labels[g_indices]
            now_correct = proj_preds[g_indices] == true_labels[g_indices]
            corrected   = g_indices[was_wrong & now_correct]
            if len(corrected) == 0:
                print(f"  Group {gid}: no corrected predictions — skipping plot.")
                continue
            top_k   = min(10, len(corrected))
            top_idx = corrected[:top_k]

            fig, axes = plt.subplots(1, top_k, figsize=(2.5 * top_k, 3.5), squeeze=False)
            fig.suptitle(
                f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — "
                f"Top {top_k} corrected after PRISM projection",
                fontsize=10, fontweight="bold",
            )
            for col in range(top_k):
                idx = top_idx[col]
                ax  = axes[0, col]
                ax.imshow(_PIL.open(test_ds.samples[idx][0]).convert("RGB"))
                ax.axis("off")
                ax.set_title(
                    f"true:{CLASS_NAMES[true_labels[idx]]}\n"
                    f"orig:{CLASS_NAMES[orig_preds[idx]]}",
                    fontsize=7,
                )
            plt.tight_layout()
            fpath = os.path.join(out_dir, f"ablation_corrected_group{gid}.png")
            plt.savefig(fpath, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"  Saved corrected plot (group {gid}): {fpath}")

            corrected_images  = [_PIL.open(test_ds.samples[i][0]).convert("RGB") for i in top_idx]
            corrected_prompts = [CLASS_NAMES[true_labels[i]] for i in top_idx]
            with torch.enable_grad():
                heatmap_results = generate_clip_heatmaps(corrected_images, corrected_prompts, clip_ft)
            fig, axes = plt.subplots(2, top_k, figsize=(2.5 * top_k, 5), squeeze=False)
            fig.suptitle(
                f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — corrected: original | heatmap",
                fontsize=10, fontweight="bold",
            )
            for col, (img_pil, hr) in enumerate(zip(corrected_images, heatmap_results)):
                axes[0, col].imshow(img_pil); axes[0, col].axis("off")
                axes[0, col].set_title(corrected_prompts[col], fontsize=7)
                axes[1, col].imshow(hr["heatmap_rgb"]); axes[1, col].axis("off")
                axes[1, col].set_title(f"sim={hr['similarity']:.2f}", fontsize=7)
            plt.tight_layout()
            hfpath = os.path.join(out_dir, f"ablation_corrected_heatmap_group{gid}.png")
            plt.savefig(hfpath, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"  Saved corrected heatmap (group {gid}): {hfpath}")

        # ── Harmed images (groups 0, 3): correct → wrong after projection ────
        for gid in [0, 3]:
            g_indices = np.where(group_ids == gid)[0]
            if len(g_indices) == 0:
                continue
            was_correct = orig_preds[g_indices] == true_labels[g_indices]
            now_wrong   = proj_preds[g_indices] != true_labels[g_indices]
            harmed      = g_indices[was_correct & now_wrong]
            if len(harmed) == 0:
                print(f"  Group {gid}: no harmed predictions — skipping plot.")
                continue
            top_k   = min(10, len(harmed))
            top_idx = harmed[:top_k]

            fig, axes = plt.subplots(1, top_k, figsize=(2.5 * top_k, 3.5), squeeze=False)
            fig.suptitle(
                f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — "
                f"Top {top_k} harmed after PRISM projection",
                fontsize=10, fontweight="bold",
            )
            for col in range(top_k):
                idx = top_idx[col]
                ax  = axes[0, col]
                ax.imshow(_PIL.open(test_ds.samples[idx][0]).convert("RGB"))
                ax.axis("off")
                ax.set_title(
                    f"true:{CLASS_NAMES[true_labels[idx]]}\n"
                    f"proj:{CLASS_NAMES[proj_preds[idx]]}",
                    fontsize=7,
                )
            plt.tight_layout()
            fpath = os.path.join(out_dir, f"ablation_harmed_group{gid}.png")
            plt.savefig(fpath, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"  Saved harmed plot (group {gid}): {fpath}")

            harmed_images  = [_PIL.open(test_ds.samples[i][0]).convert("RGB") for i in top_idx]
            harmed_prompts = [CLASS_NAMES[true_labels[i]] for i in top_idx]
            with torch.enable_grad():
                heatmap_results = generate_clip_heatmaps(harmed_images, harmed_prompts, clip_ft)
            fig, axes = plt.subplots(2, top_k, figsize=(2.5 * top_k, 5), squeeze=False)
            fig.suptitle(
                f"Group {gid} ({GROUP_NAMES.get(gid, str(gid))}) — harmed: original | heatmap",
                fontsize=10, fontweight="bold",
            )
            for col, (img_pil, hr) in enumerate(zip(harmed_images, heatmap_results)):
                axes[0, col].imshow(img_pil); axes[0, col].axis("off")
                axes[0, col].set_title(harmed_prompts[col], fontsize=7)
                axes[1, col].imshow(hr["heatmap_rgb"]); axes[1, col].axis("off")
                axes[1, col].set_title(f"sim={hr['similarity']:.2f}", fontsize=7)
            plt.tight_layout()
            hfpath = os.path.join(out_dir, f"ablation_harmed_heatmap_group{gid}.png")
            plt.savefig(hfpath, bbox_inches="tight", dpi=120)
            plt.close()
            print(f"  Saved harmed heatmap (group {gid}): {hfpath}")

        return

    else: 
        raise ValueError(f"Unknown concept_finding_method: {args.concept_finding_method}")
        
    # #     plt.show()

    if args.denoise_concepts:
        candidate_concepts, concept_weights = denois_candidate_concepts(
            candidate_concepts=candidate_concepts,
            sae_model=sae_model,
            percentile=args.denoise_percentile,
            beta=args.denoise_beta
        )

    else:
        concept_weights = None

    if args.concept_finding_method == "labelfree" :
        n_candidates = int(M.sum().item())
        print(
            f"  Candidate concepts (after denoise): {len(candidate_concepts)} concepts "
            f"over {n_candidates} candidate images"
        )
        concept_extractor_name = "ft-labelfree"
    elif args.concept_finding_method == "dialguided":
        print(
            f"  Candidate concepts (after denoise): {len(candidate_concepts)} concepts "
            f"from dial-guided spurious concept detection"
        )
        concept_extractor_name = "ft-dialguided"
    elif args.concept_finding_method == "labelguided":
        print(
            f"  Candidate concepts (after denoise): {len(candidate_concepts)} concepts "
            f"from label-guided spurious concept detection"
        )
        concept_extractor_name = "ft-labelguided"

    elif args.concept_finding_method == "none":
        print("  No concept-finding — reconstruction-only baseline (0 concepts).")
        concept_extractor_name = "ft-reconstruction_only"

    elif args.concept_finding_method == "highmag":
        print(
            f"  Candidate concepts (after denoise): {len(candidate_concepts)} concepts "
            f"from high-magnitude spurious concept detection"
        )
        concept_extractor_name = "ft-highmag"

    if args.model == "RouteSAE":
        if (len(candidate_concepts) > 0 or args.concept_finding_method == "none") and not args.skip_ablation:
            routesae_P = build_projection_matrix(
                concepts=candidate_concepts,
                sae_model=sae_model,
                method=args.projection_method,
            )
            ablation_results = {}
            for editing_method in ("deactivation", "projection"):
                result = ablate_spurious_concepts_routesae(
                    clip_ft=clip_ft,
                    sae_model=sae_model,
                    test_ds=test_ds,
                    spurious_concept_indices=candidate_concepts,
                    device=device,
                    sae_dir=sae_dir,
                    lambda_coefficient=args.ablation_coefficient,
                    vocab_names=vocab_names,
                    concept_match_scores=concept_match_scores,
                    concept_extractor_name=concept_extractor_name,
                    clip_mode=args.clip_mode,
                    report_suffix=report_suffix,
                    editing_method=editing_method,
                    P=routesae_P,
                    write_report=False,
                )
                ablation_results[editing_method] = result[1]
                orig_preds = result[0]
            _write_combined_ablation_report(
                group_ids=np.array([s[3] for s in test_ds.samples]),
                true_labels=np.array([s[1] for s in test_ds.samples]),
                orig_preds=orig_preds,
                ablation_results=ablation_results,
                concept_idx=candidate_concepts,
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                sae_dir=sae_dir,
                concept_extractor_name=concept_extractor_name,
                report_suffix=report_suffix,
            )
    else:
        if (len(candidate_concepts) > 0 or args.concept_finding_method == "none") and not args.skip_ablation:
            P = build_projection_matrix(
                concepts=candidate_concepts,
                sae_model=sae_model,
                method=args.projection_method,
            )
            ablation_results = {}
            for editing_method, ablation_hook in (
                ("deactivation", make_sae_ablation_hook(
                    sae_model, candidate_concepts, device,
                    lambda_coef=args.ablation_coefficient,
                )),
                ("projection", projection_ablation_hook(
                    P=P, lambda_coef=args.ablation_coefficient,
                )),
            ):
                result = ablate_spurious_concepts(
                    clip_ft=clip_ft,
                    sae_model=sae_model,
                    test_ds=test_ds,
                    spurious_concept_indices=candidate_concepts,
                    device=device,
                    sae_dir=sae_dir,
                    vocab_names=vocab_names,
                    concept_match_scores=concept_match_scores,
                    ablation_hook=ablation_hook,
                    lambda_coefficient=args.ablation_coefficient,
                    concept_extractor_name=concept_extractor_name,
                    candidate_mask_only=False,
                    clip_mode=args.clip_mode,
                    te_sae_reps=te_results["sae_representations"],
                    report_suffix=report_suffix,
                    write_report=False,
                )
                ablation_results[editing_method] = result[1]
                orig_preds = result[0]
            _write_combined_ablation_report(
                group_ids=np.array([s[3] for s in test_ds.samples]),
                true_labels=np.array([s[1] for s in test_ds.samples]),
                orig_preds=orig_preds,
                ablation_results=ablation_results,
                concept_idx=candidate_concepts,
                vocab_names=vocab_names,
                concept_match_scores=concept_match_scores,
                sae_dir=sae_dir,
                concept_extractor_name=concept_extractor_name,
                report_suffix=report_suffix,
            )

    # Top real ft-train images per concept (runs regardless of MACO), saved under
    # the concept-detection-technique folder — a real-image counterpart to MACO.
    if len(candidate_concepts) > 0:
        # One montage PNG per concept, so the same cap logic MACO uses applies
        # here (ranked by peak ft-train activation, the same data the montage
        # itself draws from). Kept at 50 rather than --maco_max_concepts'
        # default: montages are a quick browse, not the main artifact.
        save_concepts = top_concepts_by_activation(
            candidate_concepts, ft_results["sae_representations"],
            max_n=50, label="montage",
        )

        print(f"  Saving top-5 ft-train images for {len(save_concepts)} candidate concepts ...")
        save_top_ft_images_per_concept(
            candidate_concepts=save_concepts,
            ft_results=ft_results,
            sae_dir=sae_dir,
            concept_extractor_name=concept_extractor_name,
            clip_ft=clip_ft,
            sae_model=sae_model,
            device=device,
            vocab_names=vocab_names,
            concept_match_scores=concept_match_scores,
            top_k=5,
        )

    if args.maco_visualize_concepts and len(candidate_concepts) > 0:
        # Ranked by peak activation on the TEST split (te_results) rather than
        # ft_train: MACO synthesizes what maximally drives a concept, so the
        # concepts worth the compute are the ones that actually fire strongest
        # on the data being analyzed.
        maco_concepts = top_concepts_by_activation(
            candidate_concepts, te_results["sae_representations"],
            max_n=args.maco_max_concepts, label="MACO",
        )
        print(f"  MACO visualizing {len(maco_concepts)} candidate concepts ...")
        run_maco_parallel(
            candidate_concepts=maco_concepts,
            sae_model=sae_model,
            clip_ft=clip_ft,
            test_ds=test_ds,
            sae_dir=sae_dir,
            concept_match_scores=concept_match_scores,
            vocab_names=vocab_names,
            device=device,
            args=args,
            run_dir=run_dir,
            sae_path=sae_path,
            subfolder=concept_extractor_name,
        )

if __name__ == "__main__":
    main()
