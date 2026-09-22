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
    'load_routesae_for_clip', 'image_concepts', 'patch_concept_activations',
    'dataset_patch_activations', 'save_patch_activations', 'load_patch_activations',
    'PatchActivations', 'routesae_embeds',
    'routesae_embeds_projection', 'routesae_embeds_batched',
    'extract_routesae_representations', 'concept_layer_origins', 'self_check',
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


@torch.no_grad()
def patch_concept_activations(
    sae: RouteSAE,
    clip_model,
    pixel_values: torch.Tensor,
    concept_ids: Sequence[int],
    aggre: str = 'sum',
    routing: str = 'hard',
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-PATCH activation of the given concepts, before any pooling.

    image_concepts() max-pools RouteSAE's per-patch codes into one vector per
    image, which is what every concept selector consumes -- but that throws
    away WHERE in the image each concept fired. This returns that: the same
    latents, sliced to `concept_ids`, with the patch axis reshaped to the
    ViT's grid so it can be drawn over the image.

    Returns
    -------
    grid : (B, C, g, g)   activation of concept c at each patch, g x g grid
                          (7 x 7 for ViT-B/32 at 224px). CLS excluded.
    cls  : (B, C)         the CLS token's activation, for reference -- with
                          pool='max' the image-level value is max(cls, grid).
    Read a map as "how strongly the SAE selected concept c for each patch";
    the argmax patch is the one that set the image-level (max-pooled) value.
    """
    stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
    x, _, _ = pre_process(stack)
    _, _, latents, _, _ = sae(x, aggre, routing)          # (B, T, latent)
    idx = torch.as_tensor(list(concept_ids), device=latents.device, dtype=torch.long)
    sel = latents[:, :, idx]                              # (B, T, C)
    n_patches = sel.shape[1] - 1
    g = int(round(n_patches ** 0.5))
    if g * g != n_patches:
        raise ValueError(f"{n_patches} patches is not a square grid")
    grid = sel[:, 1:, :].permute(0, 2, 1).reshape(sel.shape[0], len(concept_ids), g, g)
    return grid, sel[:, 0, :]


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


@torch.no_grad()
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

    @torch.no_grad matches routesae_embeds above. clip_layer_stack manages
    its own no_grad, but the hooked clip_forward_last_hidden pass below did
    not -- across the tens of thousands of calls Generate_Concept_Pool makes,
    the retained graphs were enough to exhaust MPS.
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


class PatchActivations:
    """Per-patch activations of a concept list over a whole dataset, sparse.

    Dense would be (N_images, C, g*g): 5,794 x 16,309 x 49 for the full
    RouteSAE dictionary on waterbirds -- 4.6 billion floats. Almost all are
    zero (the codes are TopK-sparse), so only the non-zero entries are kept
    in COO form, which also makes the file size proportional to how many
    concepts actually fire rather than to the size of the grid.

    Attributes
    ----------
    concept_ids : (C,) int    the concepts, in the order `cpos` indexes
    image_paths : list[str]   the dataset's images, in the order `img` indexes
    grid        : int         g -- the ViT patch grid is g x g
    img, cpos, patch, value : (nnz,) arrays -- image index, position of the
                  concept in concept_ids, patch index (row-major, CLS
                  excluded), activation
    cls         : (N, C) float16  CLS-token activation per image/concept
    """

    def __init__(self, concept_ids, image_paths, grid, img, cpos, patch, value, cls):
        import numpy as np
        self.concept_ids = np.asarray(concept_ids, dtype=np.int64)
        self.image_paths = list(image_paths)
        self.grid = int(grid)
        self.img, self.cpos, self.patch, self.value = img, cpos, patch, value
        self.cls = cls
        self._pos = {int(c): i for i, c in enumerate(self.concept_ids)}
        # Sorted by (concept, image) so per-concept lookups are one slice.
        order = np.lexsort((self.img, self.cpos))
        self.img, self.cpos = self.img[order], self.cpos[order]
        self.patch, self.value = self.patch[order], self.value[order]
        self._bounds = np.searchsorted(self.cpos, np.arange(len(self.concept_ids) + 1))

    def __len__(self):
        return len(self.image_paths)

    def concept_entries(self, cid):
        """(img, patch, value) arrays of every non-zero patch of concept cid."""
        i = self._pos[int(cid)]
        a, b = self._bounds[i], self._bounds[i + 1]
        return self.img[a:b], self.patch[a:b], self.value[a:b]

    def grid_for(self, image_idx, cid):
        """Dense (g, g) map of concept cid on one image."""
        import numpy as np
        img, patch, value = self.concept_entries(cid)
        m = img == image_idx
        out = np.zeros(self.grid * self.grid, dtype=np.float32)
        out[patch[m]] = value[m]
        return out.reshape(self.grid, self.grid)

    def image_max(self, cid):
        """(N,) max over patches per image -- reproduces image_concepts(pool='max')
        up to the CLS token, which is folded in here."""
        import numpy as np
        img, _patch, value = self.concept_entries(cid)
        out = np.zeros(len(self.image_paths), dtype=np.float32)
        np.maximum.at(out, img, value)
        return np.maximum(out, self.cls[:, self._pos[int(cid)]].astype(np.float32))

    def top_patches(self, cid, k=5):
        """The k strongest (image_idx, patch_idx, value) triples of the concept
        anywhere in the dataset."""
        import numpy as np
        img, patch, value = self.concept_entries(cid)
        sel = np.argsort(-value)[:k]
        return list(zip(img[sel].tolist(), patch[sel].tolist(), value[sel].tolist()))

    def position_histogram(self, cid):
        """(g, g) count of how often each patch position holds the concept's
        per-image maximum -- flat for a content feature, peaked for a
        positional one (a concept that only ever fires on the bottom row
        shows up here immediately)."""
        import numpy as np
        img, patch, value = self.concept_entries(cid)
        hist = np.zeros(self.grid * self.grid, dtype=np.int64)
        if len(img):
            # per-image argmax: entries are sorted by img within the concept
            starts = np.r_[0, np.flatnonzero(np.diff(img)) + 1]
            ends = np.r_[starts[1:], len(img)]
            for a, b in zip(starts, ends):
                hist[patch[a + int(np.argmax(value[a:b]))]] += 1
        return hist.reshape(self.grid, self.grid)


@torch.no_grad()
def dataset_patch_activations(
    sae: RouteSAE,
    clip_model,
    dataset,
    concept_ids: Sequence[int],
    device,
    preprocess,
    batch_size: int = 64,
    concept_chunk: int = 512,
    aggre: str = 'sum',
    routing: str = 'hard',
) -> PatchActivations:
    """patch_concept_activations() swept over every image of `dataset`, kept
    sparse -- see PatchActivations for what comes back and how to read it.

    One CLIP + RouteSAE forward per batch, the same cost as
    concept_layer_origins (a pass over the split); the concept list is
    sliced in chunks of `concept_chunk` so a full-dictionary request never
    materialises a (B, 16384, 49) tensor at once.
    """
    import numpy as np
    from PIL import Image

    concept_ids = [int(c) for c in concept_ids]
    samples = dataset.samples if hasattr(dataset, 'samples') else dataset
    paths_all = [c[0] if isinstance(c, (tuple, list)) else c for c in samples]
    N, C = len(paths_all), len(concept_ids)

    img_l, cpos_l, patch_l, val_l = [], [], [], []
    cls_all = np.zeros((N, C), dtype=np.float16)
    grid_g = None

    idx_all = torch.as_tensor(concept_ids, dtype=torch.long)
    for start in range(0, N, batch_size):
        paths = paths_all[start:start + batch_size]
        pixel_values = torch.stack(
            [preprocess(Image.open(p).convert('RGB')) for p in paths]).to(device)
        # ONE forward per batch; the concept list is sliced from its output.
        # (Calling patch_concept_activations per chunk would redo the CLIP +
        # SAE forward once per chunk -- 32x the work for a full dictionary.)
        stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
        x, _, _ = pre_process(stack)
        _, _, latents, _, _ = sae(x, aggre, routing)                  # (B, T, latent)
        n_patches = latents.shape[1] - 1
        grid_g = int(round(n_patches ** 0.5))
        for c0 in range(0, C, concept_chunk):
            idx = idx_all[c0:c0 + concept_chunk].to(latents.device)
            sel = latents[:, :, idx]                                   # (B, T, c)
            flat = sel[:, 1:, :].permute(0, 2, 1).contiguous().cpu()   # (B, c, g*g)
            nz = flat.nonzero(as_tuple=False).numpy()                  # (nnz, 3)
            if len(nz):
                img_l.append(nz[:, 0] + start)
                cpos_l.append(nz[:, 1] + c0)
                patch_l.append(nz[:, 2])
                val_l.append(flat[nz[:, 0], nz[:, 1], nz[:, 2]].numpy())
            cls_all[start:start + len(paths), c0:c0 + len(idx)] = \
                sel[:, 0, :].cpu().numpy().astype(np.float16)

    cat = lambda parts, dt: (np.concatenate(parts).astype(dt) if parts
                             else np.zeros(0, dtype=dt))
    return PatchActivations(
        concept_ids=concept_ids, image_paths=paths_all, grid=grid_g or 0,
        img=cat(img_l, np.int32), cpos=cat(cpos_l, np.int32),
        patch=cat(patch_l, np.int16), value=cat(val_l, np.float32), cls=cls_all,
    )


def save_patch_activations(pa: PatchActivations, path: str) -> None:
    import numpy as np, os
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp.npz"
    np.savez_compressed(
        tmp, concept_ids=pa.concept_ids, image_paths=np.array(pa.image_paths),
        grid=np.int64(pa.grid), img=pa.img, cpos=pa.cpos, patch=pa.patch,
        value=pa.value, cls=pa.cls)
    os.replace(tmp, path)


def load_patch_activations(path: str) -> PatchActivations:
    import numpy as np
    z = np.load(path)
    return PatchActivations(
        concept_ids=z['concept_ids'], image_paths=z['image_paths'].tolist(),
        grid=int(z['grid']), img=z['img'], cpos=z['cpos'], patch=z['patch'],
        value=z['value'], cls=z['cls'])


@torch.no_grad()
def concept_layer_origins(
    sae: RouteSAE,
    clip_model,
    dataset,
    candidate_concepts: Sequence[int],
    device,
    preprocess,
    batch_size: int = 32,
    include_cls: bool = True,
    aggre: str = 'sum',
    routing: str = 'hard',
) -> Dict[int, Dict[str, object]]:
    """Which CLIP layer each concept's activations mostly come from, over `dataset`.

    Unlike a per-layer SAE, RouteSAE shares one dictionary across layers and
    routes each PATCH to a layer per forward pass (routesae.py's
    RouteSAE.get_router_weights/get_sae_input) -- "layer" isn't a fixed
    attribute of a concept id the way it would be for a per-layer dictionary.
    Under hard routing (the default everywhere in this project) each patch is
    still routed to exactly one layer per pass, so this measures it
    empirically: for every image where a candidate concept is active (its
    strongest patch activation is > 0), look up which layer that patch was
    routed to (routesae.routed_layers' formula, inlined here to reuse one
    forward pass instead of two), and tally. Returns, per concept:

        layer_counts     : {layer: n_images} for every layer with >0 counts
        mode_layer       : the single most common layer (None if never active)
        purity           : mode_layer's share of all active images -- low
                            purity means the concept doesn't have a stable
                            single-layer origin, which is worth surfacing
                            rather than silently picking the mode anyway
        n_active_images  : how many images (out of len(dataset)) the concept
                            was active in at all

    Raises ValueError for routing='soft': soft routing blends every layer
    into each patch, so there is no single origin layer to attribute.
    """
    from PIL import Image

    if routing != 'hard':
        raise ValueError(
            "concept_layer_origins only makes sense for routing='hard' -- "
            "'soft' routing mixes every layer into each patch, so no single "
            "origin layer exists to attribute a concept to."
        )

    cand = list(candidate_concepts)
    cand_t = torch.tensor(cand, device=device, dtype=torch.long)
    counts = torch.zeros(len(cand), sae.n_routed_layers, dtype=torch.long)
    n_active = torch.zeros(len(cand), dtype=torch.long)

    samples = dataset.samples if hasattr(dataset, 'samples') else dataset
    for start in range(0, len(samples), batch_size):
        chunk = samples[start:start + batch_size]
        paths = [c[0] if isinstance(c, (tuple, list)) else c for c in chunk]
        imgs = [preprocess(Image.open(p).convert('RGB')) for p in paths]
        pixel_values = torch.stack(imgs).to(device)

        stack = clip_layer_stack(clip_model, pixel_values, sae.n_layers)
        x, _, _ = pre_process(stack)
        _, _, latents, _, router_weights = sae(x, aggre, routing)
        # 0-based, relative to sae.start_layer -- routesae.routed_layers adds
        # start_layer to report true CLIP layer numbers; done below instead,
        # once per concept, after aggregating.
        layer_per_patch = router_weights.argmax(dim=-1)          # (B, T)

        patches = latents if include_cls else latents[:, 1:, :]
        layers  = layer_per_patch if include_cls else layer_per_patch[:, 1:]

        cand_latents = patches[:, :, cand_t]                     # (B, T, n_cand)
        max_vals, max_patch = cand_latents.max(dim=1)             # (B, n_cand)
        origin_layer = layers.gather(1, max_patch)                # (B, n_cand)

        active = (max_vals > 0).cpu()
        origin_layer = origin_layer.cpu()
        for b in range(active.size(0)):
            active_cols = active[b].nonzero(as_tuple=True)[0]
            if active_cols.numel() == 0:
                continue
            for ci, lyr in zip(active_cols.tolist(), origin_layer[b, active_cols].tolist()):
                counts[ci, lyr] += 1
                n_active[ci] += 1

    results: Dict[int, Dict[str, object]] = {}
    for i, cid in enumerate(cand):
        row, total = counts[i], int(n_active[i].item())
        if total == 0:
            results[cid] = dict(layer_counts={}, mode_layer=None, purity=0.0, n_active_images=0)
            continue
        mode_idx = int(row.argmax().item())
        results[cid] = dict(
            layer_counts={sae.start_layer + j: int(c) for j, c in enumerate(row.tolist()) if c > 0},
            mode_layer=sae.start_layer + mode_idx,
            purity=int(row[mode_idx].item()) / total,
            n_active_images=total,
        )
    return results


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
