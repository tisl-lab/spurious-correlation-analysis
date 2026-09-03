"""
Concept naming for RouteSAE, following msae/sae_naming.py.

Produces the same artifact the ftclip_msae pipeline already consumes:

    Concept_Interpreter_<sae_stem>_<vocab_stem>.npy   shape (n_vocab, n_latents)

so concepts are named exactly as for MSAE:

    vocab_names[concept_match_scores[:, cid].argmax()]

Method, matching sae_naming.compute_similarities:
    decoded_search_space = decoder + pre_bias
    if patch_diff: subtract decode(zeros)         # removes the constant offset
    scores = cosine_similarity(vocab_text_embeddings, decoded_search_space)

One extra step is unavoidable. MSAE's decoder already lives in CLIP's 512-d
joint space, so it can be compared with text directly. RouteSAE's decoder lives
in the 768-d residual stream at layers n/4..3n/4, so each direction is pushed
through CLIP's output head (ln_post + visual projection) to reach the joint
space. That skips the layers above the concept, making these names a strong
hint rather than a causal measurement - use --verify to check a few by steering.

Usage:
    python routesae_naming.py \\
        -m routesae_weights/routesae_K32_ViT-B~32_16384.pt \\
        -v results/waterbirds/embeddings/waterbirds_domain_ViT-B~32_zs_-1_text_954_512.npy \\
        -p results/waterbirds/embeddings/concept_match/RouteSAE \\
        --clip ViT-B/32
"""

import argparse
import logging
import os

import numpy as np
import torch
import torch.nn.functional as F

from routesae import RouteSAE, load_routesae, clip_backend, clip_embeds_with_sae
from routesae_train import load_clip, clip_dims, resolve_device

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description='Compute concept-match scores for a RouteSAE')
    p.add_argument('-m', '--model', required=True, help='Trained RouteSAE .pt')
    p.add_argument('-v', '--vocab', required=True,
                   help='Vocabulary CLIP text embeddings .npy, shape (n_vocab, 512)')
    p.add_argument('-p', '--path-to-save', default='.', help='Directory for the .npy')
    p.add_argument('--clip', required=True,
                   help="CLIP the SAE was trained on ('ViT-B/32' or a fine-tuned .pt)")
    p.add_argument('--latent-size', type=int, default=16384)
    p.add_argument('-k', type=int, default=32)
    p.add_argument('--patch-diff', default=True, action='store_true',
                   help='Subtract decode(0), as in sae_naming.py')
    p.add_argument('--logit-scale', type=float, default=1.0)
    p.add_argument('--device', default='auto')
    p.add_argument('--verbose', action='store_true', help='Print the top names found')
    p.add_argument('--names', default=None,
                   help='Companion vocab .txt (defaults to the sibling file, dim suffix dropped)')
    p.add_argument('--verify', type=int, default=0,
                   help='Causally verify the top N concepts by steering (needs --data)')
    p.add_argument('--data', default=None, help='Waterbirds root, for --verify')
    return p.parse_args()


def cosine_similarity_matrix(A: torch.Tensor, B: torch.Tensor,
                             logit_scale: float = 1.0) -> torch.Tensor:
    """Cosine similarity between every row of A and every row of B."""
    A_normalized = A / A.norm(dim=1, keepdim=True)
    B_normalized = B / B.norm(dim=1, keepdim=True)
    return logit_scale * A_normalized @ B_normalized.t()


def decoder_search_space(sae: RouteSAE, clip_model, patch_diff: bool = True) -> torch.Tensor:
    """RouteSAE decoder directions, projected into CLIP's joint space.

    Mirrors sae_naming.compute_similarities, plus the projection RouteSAE needs.
    """
    inner = sae.sae                                        # the TopK SAE inside RouteSAE
    directions = inner.decoder.weight.T + inner.pre_bias   # (latent, hidden)

    if patch_diff:
        zero_space = inner.decode(torch.zeros(1, inner.latent_size,
                                              dtype=directions.dtype,
                                              device=directions.device))
        directions = directions - zero_space

    # Residual stream (768) -> joint space (512), via CLIP's own output head.
    if clip_backend(clip_model) == 'openai':
        visual = clip_model.visual
        projected = visual.ln_post(directions)
        if visual.proj is not None:
            projected = projected @ visual.proj
    else:
        projected = clip_model.vision_model.post_layernorm(directions)
        projected = clip_model.visual_projection(projected)

    return projected.float()


def load_vocab_embeddings(path: str) -> np.ndarray:
    """Load vocabulary embeddings the way msae/utils.py SAEDataset does.

    These files are raw float32 memmaps, not pickled arrays: the shape comes
    from the last two underscore-separated numbers in the filename, e.g.
    ..._text_954_512.npy -> (954, 512).
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    parts = stem.split('_')
    try:
        n_rows, vector_size = int(parts[-2]), int(parts[-1])
    except (IndexError, ValueError):
        logger.info('Filename carries no _<n>_<dim> suffix; trying np.load')
        return np.asarray(np.load(path))

    logger.info(f'Reading memmap {path} as ({n_rows}, {vector_size})')
    return np.array(np.memmap(path, dtype='float32', mode='r',
                              shape=(n_rows, vector_size)))


def default_names_path(vocab_path: str) -> str:
    """Companion .txt for a vocab .npy: same stem with the trailing _<dim> dropped."""
    import re
    stem = os.path.splitext(os.path.basename(vocab_path))[0]
    return os.path.join(os.path.dirname(vocab_path), re.sub(r'_\d+$', '', stem) + '.txt')


def main() -> None:
    args = parse_args()
    device = resolve_device(args.device)

    _, clip_model = load_clip(args.clip, device)
    hidden_size, n_layers = clip_dims(clip_model)
    sae = load_routesae(args.model, hidden_size, n_layers, args.latent_size, args.k, device)
    logger.info(f'Loaded RouteSAE: latent={sae.latent_size} hidden={hidden_size} k={sae.k}')

    vocab = torch.from_numpy(load_vocab_embeddings(args.vocab)).float().to(device)
    logger.info(f'Vocabulary embeddings: {tuple(vocab.shape)}')

    with torch.no_grad():
        search_space = decoder_search_space(sae, clip_model, args.patch_diff)
        if search_space.shape[1] != vocab.shape[1]:
            raise ValueError(
                f'Dimension mismatch: projected decoder is {search_space.shape[1]}-d but '
                f'the vocabulary is {vocab.shape[1]}-d. Use a vocabulary embedded with the '
                f'same CLIP variant.'
            )
        scores = cosine_similarity_matrix(vocab, search_space, args.logit_scale)

    scores_np = scores.cpu().numpy().astype(np.float32)
    assert scores_np.shape == (vocab.shape[0], sae.latent_size)

    os.makedirs(args.path_to_save, exist_ok=True)
    sae_stem = os.path.splitext(os.path.basename(args.model))[0]
    vocab_stem = os.path.splitext(os.path.basename(args.vocab))[0]
    out_path = os.path.join(args.path_to_save,
                            f'Concept_Interpreter_{sae_stem}_{vocab_stem}.npy')
    np.save(out_path, scores_np)
    logger.info(f'Saved {scores_np.shape} concept-match scores to {out_path}')

    names_path = args.names or default_names_path(args.vocab)
    if not os.path.isfile(names_path):
        logger.warning(f'Companion names file not found: {names_path}')
        return

    with open(names_path) as f:
        vocab_names = [line.strip() for line in f if line.strip()]
    if len(vocab_names) != scores_np.shape[0]:
        logger.warning(f'names ({len(vocab_names)}) != score rows ({scores_np.shape[0]})')
        return

    # Name every concept the same way the msae pipeline does.
    best_row = scores_np.argmax(axis=0)
    best_val = scores_np.max(axis=0)
    order = np.argsort(-best_val)[:20]

    logger.info('Strongest concept matches:')
    for cid in order:
        logger.info(f'  concept {cid:6}  {vocab_names[best_row[cid]]:32} '
                    f'score {best_val[cid]:.4f}')

    if args.verify and args.data:
        from routesae_train import make_loader
        _, clip_model2 = load_clip(args.clip, device)
        preprocess, _ = load_clip(args.clip, device)
        loader = make_loader(args.data, preprocess, 16, 'val', 16)
        pixel_values = next(iter(loader))[0].to(device)

        logger.info('Causal check (steering) on the top concepts:')
        with torch.no_grad():
            base = clip_embeds_with_sae(clip_model, sae, pixel_values)
            for cid in order[:args.verify]:
                amped = clip_embeds_with_sae(clip_model, sae, pixel_values,
                                             amplify=[(int(cid), 8.0)])
                delta = (amped @ vocab.T).mean(0) - (base @ vocab.T).mean(0)
                top = delta.topk(3).indices.tolist()
                logger.info(f'  concept {cid:6}  lens={vocab_names[best_row[cid]]:24} '
                            f'steer={[vocab_names[t] for t in top]}')


if __name__ == '__main__':
    main()
