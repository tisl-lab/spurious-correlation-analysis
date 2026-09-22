"""Generate case-averaged domain-vocab text embeddings for a fine-tuned CLIP.

For every word in msae/vocab/waterbirds_domain_vocab.txt we embed BOTH the
lowercase form ("eagle") and the first-letter-capitalized form ("Eagle") through
the model's (frozen) text encoder and AVERAGE the two vectors — one embedding per
concept. This makes the concept-naming vocabulary case-robust.

precompute_activations.py cannot do this averaging, so we do it here. The output
is byte-for-byte compatible with precompute's format: raw float32 C-order binary
named  waterbirds_domain_<TAG>_ft400_-1_text_<N>_<DIM>.npy  (SAEDataset parses N
and DIM from the filename and memmaps it), plus a companion  ..._text_<N>.txt
holding the row-aligned words.

Usage:
    python tools/gen_domain_caseavg.py <model.pt> <TAG> <out_dir> <DIM>
      <model.pt>  fine-tuned CLIP checkpoint (run_dir/model.pt)
      <TAG>       model tag used in filenames, e.g. ViT-B~32 or ViT-L~14
      <out_dir>   embeddings folder to write into (that model's own folder)
      <DIM>       embedding dim (512 for ViT-B/32, 768 for ViT-L/14)

    Zero-shot mode (no fine-tuned checkpoint -- loads pretrained CLIP
    directly, and matches precompute_activations.py's own "_zs" filename
    convention instead of "_ft400"):
    python tools/gen_domain_caseavg.py --zeroshot <clip_model> <TAG> <out_dir> <DIM>
      <clip_model>  e.g. "ViT-L/14" (passed to CLIPZeroShot(model_name=...))
"""
import os
import sys

import numpy as np
import torch

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)
import clip                              # noqa: E402
from clip_zero_shot import CLIPZeroShot  # noqa: E402

VOCAB = os.path.join(REPO, "msae", "vocab", "waterbirds_domain_vocab.txt")


def capitalize(w):
    return w[:1].upper() + w[1:]


def main(model_ref, tag, out_dir, dim, zeroshot=False):
    device = ("cuda" if torch.cuda.is_available()
              else "mps" if torch.backends.mps.is_available() else "cpu")
    if zeroshot:
        m = CLIPZeroShot(model_name=model_ref, device=device)
        suffix = "_zs"
    else:
        m = CLIPZeroShot.load_model(model_ref, device=device)
        suffix = "_ft400"
    m.model.eval()

    words = [l.strip() for l in open(VOCAB) if l.strip()]
    n = len(words)
    embs = np.zeros((n, dim), dtype=np.float32)

    with torch.no_grad():
        for i in range(0, n, 256):
            chunk = words[i:i + 256]
            low = m.model.encode_text(clip.tokenize(chunk).to(device)).float()
            cap = m.model.encode_text(clip.tokenize([capitalize(w) for w in chunk]).to(device)).float()
            embs[i:i + len(chunk)] = ((low + cap) / 2.0).cpu().numpy().astype(np.float32)

    base = f"waterbirds_domain_{tag}{suffix}_-1_text_{n}_{dim}"
    embs.tofile(os.path.join(out_dir, base + ".npy"))
    with open(os.path.join(out_dir, f"waterbirds_domain_{tag}{suffix}_-1_text_{n}.txt"), "w") as f:
        f.write("\n".join(words))
    print(f"saved {base}.npy  shape=({n}, {dim})  + names txt")


if __name__ == "__main__":
    argv = sys.argv[1:]
    zeroshot = "--zeroshot" in argv
    if zeroshot:
        argv.remove("--zeroshot")
    if len(argv) != 4:
        sys.exit(__doc__)
    main(argv[0], argv[1], argv[2], int(argv[3]), zeroshot=zeroshot)
