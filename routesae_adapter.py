"""
Adapter exposing RouteSAE through the interfaces the ftclip_msae project already
uses for MSAE.

The two SAEs sit at different places in CLIP, which is the whole reason this
file exists:

    MSAE      final joint embedding      (B, 512)     1 vector  per image
    RouteSAE  residual stream, layers    (B, 50, 768) 50 vectors per image
              n/4 .. 3n/4 (7 for ViT-B/32)

So RouteSAE cannot use a single forward hook on CLIP's visual output. It hooks
each routed layer and masks to the patches routed there. This module hides that
difference behind calls shaped like the MSAE ones.

Requires: routesae.py (portable module) next to this file.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn.functional as F

from routesae import (
    RouteSAE, load_routesae, pre_process, clip_layer_stack,
    clip_image_embeds, clip_forward_last_hidden, clip_embeds_original,
    clip_backend, clip_num_layers, hook_routesae, hook_routesae_batched,
    hook_routesae_projection,
)

__all__ = [
    'load_routesae_for_clip', 'image_concepts', 'routesae_embeds',
    'routesae_embeds_projection', 'routesae_embeds_batched',
    'extract_routesae_representations', 'self_check',
]


def load_routesae_for_clip(checkpoint: str, device='cpu', **kw) -> RouteSAE:
    """Load a RouteSAE checkpoint (defaults are ViT-B/32: 768/12/16384/32)."""
    return load_routesae(checkpoint, device=device, **kw)


# ---------------------------------------------------------------------------
# Concepts
# ---------------------------------------------------------------------------

@torch.no_grad()
def image_concepts(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    pool: str = 'max',
    include_cls: bool = True,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """Image-level concept vector, (B, latent_size).

    RouteSAE produces one sparse code per patch; MSAE produces one per image.
    Pooling over patches puts them in the same shape so downstream concept
    selection code works unchanged.

    Args:
        pool: 'max'  - strongest activation of each concept anywhere in the image
                       (concept is present if any patch shows it; recommended)
              'sum'  - total evidence, favours concepts spread over many patches
              'mean' - sum divided by patch count
              'cls'  - use only the CLS patch, closest to MSAE's image-level view
        include_cls: include position 0 when pooling over patches
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    _, _, latents, _, _ = sae(x, aggre, routing)      # (B, T, latent)

    if pool == 'cls':
        return latents[:, 0, :]

    patches = latents if include_cls else latents[:, 1:, :]
    if pool == 'max':
        return patches.max(dim=1).values
    elif pool == 'sum':
        return patches.sum(dim=1)
    elif pool == 'mean':
        return patches.mean(dim=1)
    raise ValueError(f"pool must be one of 'max', 'sum', 'mean', 'cls'; got {pool}")


# ---------------------------------------------------------------------------
# CLIP embeddings, with optional concept removal
# ---------------------------------------------------------------------------

@torch.no_grad()
def routesae_embeds(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    concept_idx: Optional[Sequence[int]] = None,
    lambda_coef: float = 1.0,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """CLIP image embeddings computed through RouteSAE, (B, 512), L2-normalized.

    Mirrors make_sae_ablation_hook's semantics: latents for concept_idx are
    scaled by (1 - lambda_coef), so lambda_coef=1.0 removes a concept entirely
    and 0.0 leaves it untouched. Pass concept_idx=None for a plain
    reconstruction with no edit.
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    batch_layer_weights, _, _, _, _ = sae(x, aggre, routing)

    edits = None
    if concept_idx:
        scale = 1.0 - lambda_coef
        edits = [(int(i), float(scale), 1) for i in concept_idx]   # mode 1 = multiply

    handles = hook_routesae(
        sae, clip_model, batch_layer_weights,
        set_high=edits, aggre=aggre, routing=routing,
    )
    try:
        return clip_image_embeds(clip_model, clip_forward_last_hidden(clip_model, pixel_values))
    finally:
        for h in handles:
            h.remove()


@torch.no_grad()
def routesae_embeds_projection(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    P: torch.Tensor,
    lambda_coef: float = 1.0,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """CLIP image embeddings computed through RouteSAE with projection
    ablation, (B, 512), L2-normalized.

    Unlike routesae_embeds (which scales the concepts' SAE latents and
    decodes), this projects the routed residual stream directly onto the
    orthogonal complement of span(P) -- a precomputed (hidden_size,
    hidden_size) projection matrix covering the concept directions to
    remove, built by msae_ftclip.build_projection_matrix() (QR or pinv
    method, see its docstring) from the same decoder rows routesae_embeds's
    concept_idx would scale. No SAE encode is needed since the directions
    are already known.

    lambda_coef=1.0 fully removes the subspace, 0.0 leaves it untouched.

    Pass an empty/None P for "0 concepts ablated" -- note this is NOT the
    same value as routesae_embeds(concept_idx=None): that still round-trips
    through encode/decode (so it carries the SAE's own reconstruction
    error), while an empty P here registers no hook at all, i.e. the raw
    CLIP embedding. There is no SAE round-trip in the projection method to
    begin with (the whole point is skipping it), so "no concepts to project
    out" means no intervention whatsoever, not "reconstruct with nothing
    edited."
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    batch_layer_weights, _, _, _, _ = sae(x, aggre, routing)

    if P is None or P.numel() == 0:
        handles = []
    else:
        handles = hook_routesae_projection(
            sae, clip_model, batch_layer_weights, P=P, lambda_coef=lambda_coef,
        )
    try:
        return clip_image_embeds(clip_model, clip_forward_last_hidden(clip_model, pixel_values))
    finally:
        for h in handles:
            h.remove()


def routesae_embeds_batched(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    sample_concept_idx: torch.Tensor,
    aggre: str = 'sum',
    routing: str = 'hard'
) -> torch.Tensor:
    """Like routesae_embeds, but zeros a DIFFERENT single concept per sample
    in the batch (sample_concept_idx, shape (B,)) instead of one shared
    concept_idx list applied to every row.

    Lets many unrelated (concept, image) pairs -- e.g. testing whether
    ablating concept A flips image X's prediction, and concept B flips
    image Y's, in totally different images -- share ONE forward pass instead
    of one pass per concept. Used by msae_ftclip.py's Generate_Concept_Pool
    to batch the exhaustive "does zeroing concept c flip any prediction"
    search across all active concepts, rather than looping one concept at a
    time (each of which re-pays the router/layer-selection forward pass).
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    batch_layer_weights, _, _, _, _ = sae(x, aggre, routing)

    handles = hook_routesae_batched(
        sae, clip_model, batch_layer_weights,
        sample_concept_idx=sample_concept_idx, aggre=aggre, routing=routing,
    )
    try:
        return clip_image_embeds(clip_model, clip_forward_last_hidden(clip_model, pixel_values))
    finally:
        for h in handles:
            h.remove()


# ---------------------------------------------------------------------------
# Drop-in for extract_sae_representations
# ---------------------------------------------------------------------------

@torch.no_grad()
def extract_routesae_representations(
    clip_model,
    sae: RouteSAE,
    dataset,
    device,
    preprocess,
    pool: str = 'max',
    batch_size: int = 32
) -> Dict[str, object]:
    """Same return contract as the project's extract_sae_representations.

    Keys: clip_representations, sae_representations, sae_reconstructed,
    image_paths, metrics.

    Note 'sae_reconstructed' here is CLIP's *embedding* recomputed with the SAE
    in the loop, not a direct decode of the input - RouteSAE reconstructs
    intermediate activations, so its effect is only observable after the
    remaining CLIP layers run.
    """
    from PIL import Image

    clip_reps, sae_reps, sae_recons, image_paths, metrics_all = [], [], [], [], []
    samples = dataset.samples if hasattr(dataset, 'samples') else dataset

    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        paths = [c[0] if isinstance(c, (tuple, list)) else c for c in chunk]
        imgs = [preprocess(Image.open(p).convert('RGB')) for p in paths]
        pixel_values = torch.stack(imgs).to(device)

        emb_orig = clip_embeds_original(clip_model, pixel_values)
        emb_recon = routesae_embeds(sae, clip_model, pixel_values)
        concepts = image_concepts(sae, clip_model, pixel_values, pool=pool)

        clip_reps.append(emb_orig.cpu())
        sae_recons.append(emb_recon.cpu())
        sae_reps.append(concepts.cpu())
        image_paths.extend(paths)

        cs = F.cosine_similarity(emb_orig, emb_recon, dim=-1)
        l0 = (concepts != 0).sum(dim=-1).float()
        for i in range(len(paths)):
            metrics_all.append({
                'cs': cs[i].item(),
                'l0': l0[i].item(),
                'highest_magnitude': concepts[i].max().item(),
            })

    return {
        'clip_representations': torch.cat(clip_reps),
        'sae_representations': torch.cat(sae_reps),
        'sae_reconstructed': torch.cat(sae_recons),
        'image_paths': image_paths,
        'metrics': metrics_all,
    }


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------

@torch.no_grad()
def self_check(sae: RouteSAE, clip_model, pixel_values: torch.Tensor) -> Dict[str, float]:
    """Sanity checks to run before trusting any comparison.

    Returns cosine similarity between original and reconstructed embeddings,
    the sparsity actually observed, and the effect of ablating the top concept.
    A cos_recon far below ~0.9, or l0 != k, means something is wired wrong -
    most often missing pre_process normalization or a mismatched checkpoint.
    """
    emb_orig = clip_embeds_original(clip_model, pixel_values)
    emb_recon = routesae_embeds(sae, clip_model, pixel_values)
    concepts = image_concepts(sae, clip_model, pixel_values, pool='max')
    top = concepts.sum(dim=0).argmax().item()
    emb_abl = routesae_embeds(sae, clip_model, pixel_values, concept_idx=[top])

    return {
        'clip_backend': clip_backend(clip_model),
        'clip_layers': clip_num_layers(clip_model),
        'cos_original_vs_reconstructed': F.cosine_similarity(emb_orig, emb_recon, dim=-1).mean().item(),
        'cos_reconstructed_vs_ablated': F.cosine_similarity(emb_recon, emb_abl, dim=-1).mean().item(),
        'mean_l0_per_patch': float(sae.k),
        'concepts_alive_in_batch': int((concepts != 0).any(dim=0).sum()),
        'top_concept': top,
        'embedding_dim': emb_orig.shape[-1],
    }
