#!/usr/bin/env python3
"""Compute layer-routing usage percentages for an already-trained RouteSAE
checkpoint, without retraining or touching the checkpoint file.

routesae_train.py computes this exact histogram during training (see its
`layer_hist` in train()) but only prints it via logger.info -- it's never
saved to routesae_results/*.json. This script recomputes the same statistic
by running one read-only forward pass over a data split with the SAE's
router, for checkpoints (like the K=32 ones trained before this addition
existed) where that training-time log wasn't captured anywhere.

Usage:
    python tools/routesae_layer_routing.py \
        --clip ViT-B/32 --sae routesae_weights/routesae_K32_ViT-B~32_16384.pt \
        --data data/waterbirds --k 32
"""
import argparse
import os
import sys

import torch

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from routesae_train import load_clip, clip_dims, make_loader  # noqa: E402
from routesae import load_routesae, pre_process, clip_layer_stack  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--clip", required=True, help="ViT-B/32, a fine-tuned .pt, or an HF id")
    ap.add_argument("--sae", required=True, help="Trained RouteSAE checkpoint (.pt)")
    ap.add_argument("--data", required=True, help="Waterbirds root")
    ap.add_argument("--latent_size", type=int, default=16384)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--split", default="train", choices=["train", "val", "test"],
                     help="Which split to route over (train matches what training itself saw)")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--aggre", default="sum", choices=["sum", "mean"])
    ap.add_argument("--routing", default="hard", choices=["hard", "soft"])
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")

    preprocess, clip_model = load_clip(args.clip, device)
    hidden_size, n_layers = clip_dims(clip_model)
    sae = load_routesae(args.sae, hidden_size, n_layers, args.latent_size, args.k, device)
    sae.eval()

    loader = make_loader(args.data, preprocess, args.batch_size, args.split, args.limit, shuffle=False)
    layer_hist = torch.zeros(sae.n_routed_layers)

    with torch.no_grad():
        for batch in loader:
            pixel_values = batch[0].to(device)
            stack = clip_layer_stack(clip_model, pixel_values, n_layers)
            x, _, _ = pre_process(stack)
            blw, _, _, _, _ = sae(x, args.aggre, args.routing)
            layer_hist += blw.sum(dim=(0, 1)).detach().float().cpu()

    total = layer_hist.sum()
    print(f"Layer routing over {args.split} patches ({args.sae}):")
    for j, v in enumerate(layer_hist):
        print(f"  layer {sae.start_layer + j:2}: {100 * v / total:5.1f}%")


if __name__ == "__main__":
    main()
