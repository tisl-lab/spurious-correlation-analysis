"""
Standalone RouteSAE training and evaluation for CLIP ViT — no dependency on the
RouteSAE repo. Requires only routesae.py next to this file.

Works with either CLIP backend:
    --clip ViT-B/32                     stock OpenAI CLIP
    --clip /path/to/finetuned/model.pt  your fine-tuned OpenAI CLIP checkpoint
    --clip openai/clip-vit-base-patch32 HuggingFace

Usage:
    python routesae_train.py --dataset waterbirds --clip ViT-B/32 --data ./data/waterbirds --epochs 10
    python routesae_train.py --dataset celeba --clip ViT-B/32 --data ./data --epochs 10
    python routesae_train.py --clip results/.../model.pt --data ... --eval_split test
    python routesae_train.py --clip ... --data ... --eval_only --sae routesae_weights/x.pt

    # CelebA only -- train on just the images the CLIP was fine-tuned on:
    python routesae_train.py --dataset celeba --clip results/celeba/<run>/model.pt \
        --data ./data --ft_manifest results/celeba/<run>/ft_train_manifest.csv --epochs 200
"""

import argparse
import csv
import json
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Dataset

from routesae import (
    RouteSAE, load_routesae, pre_process, clip_backend, clip_num_layers,
    clip_layer_stack, clip_image_embeds, clip_forward_last_hidden,
    clip_embeds_original, hook_routesae,
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger('routesae')

SPLIT_CODES = {'train': 0, 'val': 1, 'test': 2}
# Per-dataset prompts and group names come from dataset_settings.DatasetSpec
# (--dataset), so adding a dataset means adding a spec, not editing this file.
# The two module-level constants below are the waterbirds fallback kept for
# callers that import them; _spec_for() overrides both at runtime.
ZERO_SHOT_PROMPTS = ['a photo of a landbird', 'a photo of a waterbird']
GROUP_NAMES = {
    (0, 0): 'landbird on land',
    (0, 1): 'landbird on water',
    (1, 0): 'waterbird on land',
    (1, 1): 'waterbird on water',
}


# ---------------------------------------------------------------------------
# CLIP loading
# ---------------------------------------------------------------------------

def load_clip(spec: str, device):
    """Load CLIP from an arch name, a fine-tuned .pt, or a HuggingFace id.

    Returns (preprocess, model). The model is frozen and in eval mode.
    """
    if spec.endswith(('.pt', '.pth')) or spec.startswith(('ViT-', 'RN')):
        import clip as openai_clip

        if os.path.isfile(spec):
            blob = torch.load(spec, map_location='cpu', weights_only=False)
            state_dict = blob.get('state_dict', blob) if isinstance(blob, dict) else blob
            arch = blob.get('model_name', 'ViT-B/32') if isinstance(blob, dict) else 'ViT-B/32'
            logger.info(f'Loading fine-tuned OpenAI CLIP ({arch}) from {spec}')
            model, preprocess = openai_clip.load(arch, device='cpu')
            model = model.float()
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing or unexpected:
                logger.warning(f'{len(missing)} missing / {len(unexpected)} unexpected keys')
        else:
            logger.info(f'Loading stock OpenAI CLIP: {spec}')
            model, preprocess = openai_clip.load(spec, device='cpu')
            model = model.float()
    else:
        from transformers import CLIPModel, CLIPProcessor
        logger.info(f'Loading HuggingFace CLIP: {spec}')
        model = CLIPModel.from_pretrained(spec)
        processor = CLIPProcessor.from_pretrained(spec)
        preprocess = lambda im: processor(images=im, return_tensors='pt')['pixel_values'][0]
        model._hf_processor = processor

    model = model.to(device).eval()
    model.requires_grad_(False)
    logger.info(f'CLIP ready on {device}: backend={clip_backend(model)}, '
                f'layers={clip_num_layers(model)}')
    return preprocess, model


def clip_dims(clip_model) -> Tuple[int, int]:
    """(hidden_size, n_layers) of the vision tower."""
    if clip_backend(clip_model) == 'openai':
        return clip_model.visual.conv1.out_channels, len(clip_model.visual.transformer.resblocks)
    cfg = clip_model.config.vision_config
    return cfg.hidden_size, cfg.num_hidden_layers


@torch.no_grad()
def text_class_embeddings(clip_model, device, prompts=None) -> torch.Tensor:
    """Normalized text embeddings for the zero-shot prompts."""
    prompts = prompts or ZERO_SHOT_PROMPTS

    if clip_backend(clip_model) == 'openai':
        import clip as openai_clip
        tokens = openai_clip.tokenize(prompts).to(device)
        return F.normalize(clip_model.encode_text(tokens).float(), dim=-1)

    processor = clip_model._hf_processor
    inputs = processor(text=prompts, return_tensors='pt', padding=True).to(device)
    embeds = clip_model.get_text_features(**inputs)
    if hasattr(embeds, 'pooler_output'):
        embeds = embeds.pooler_output
    return F.normalize(embeds.float(), dim=-1)


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

def _spec_for(name: str):
    """DatasetSpec for --dataset, with its prompts/groups installed as this
    module's globals so every downstream reference picks them up."""
    global ZERO_SHOT_PROMPTS, GROUP_NAMES
    import dataset_settings
    spec = dataset_settings.get(name)
    ZERO_SHOT_PROMPTS = spec.zero_shot_prompts()
    GROUP_NAMES = {pair: spec.group_name(*pair) for pair in spec.group_pairs()}
    return spec


def _read_ft_manifest(path: str) -> List[Tuple[str, int, int]]:
    """Read a run dir's ft_train_manifest.csv -> [(img_path, y, place)].

    USED FOR CELEBA ONLY, and only via --ft_manifest. The waterbirds RouteSAEs
    (both zero-shot and fine-tuned) were trained on the dataset's whole train
    split -- see batch_files/run_routesae_prep_server.sh, whose two training
    calls differ only in --clip. Reproducing that on CelebA would mean 162,770
    images against waterbirds' 4,795, so for CelebA the fine-tuned SAE is
    trained on the 800 images its CLIP was actually fine-tuned on instead.

    The trade-off, recorded here because the checkpoint name does not carry it:
    a biased fine-tune set is two groups only (aligned), so an SAE trained on
    it never sees the misaligned groups it is later asked to explain, and
    800 images at the default 10 epochs is 250 optimizer steps -- raise
    --epochs to keep the step count in the same range as a full-split run.

    Paths in the manifest are relative to the repo root (data/celeba/...), so
    run from there; `root` is not prepended.
    """
    records = []
    with open(path, newline='', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            label = int(row['label'])
            # group_id = label * 2 + attribute, so the attribute comes back out
            # by subtraction. Only used for the group stats, never for training.
            place = int(row['group_id']) - 2 * label
            records.append((row['img_path'], label, place))
    if not records:
        raise ValueError(f'{path} holds no rows')
    if not os.path.exists(records[0][0]):
        raise FileNotFoundError(
            f"First image in {path} not found at {records[0][0]!r}. Manifest "
            f"paths are relative to the repo root -- run this from there.")
    return records


class MetadataDataset(Dataset):
    """One split of any dataset that has a waterbirds-style metadata.csv
    (img_filename, y, split, place); images decoded lazily.

    Waterbirds ships that file; CelebA's loader writes an identical one on
    first use (datasets/celeba.build_metadata), so both are read by the same
    code. `root` is where the CSV and the image paths it holds are rooted --
    DatasetSpec.metadata_path() resolves it from --data."""

    def __init__(self, root: str, preprocess, split: str = 'train',
                 limit: Optional[int] = None, name: str = 'dataset',
                 manifest: Optional[str] = None):
        if manifest:
            # CelebA only -- see _read_ft_manifest. The split is ignored: the
            # manifest IS the image list.
            self.root, self.preprocess = '', preprocess
            self.records = _read_ft_manifest(manifest)
            if limit:
                self.records = self.records[:limit]
            logger.info(f'{name} ft_manifest: {len(self.records)} images '
                        f'from {manifest}')
            return

        metadata = os.path.join(root, 'metadata.csv')
        if not os.path.isfile(metadata):
            raise FileNotFoundError(
                f'metadata.csv not found in {root}\n'
                f'For CelebA it is written on the first datasets.CelebA(...) call; '
                f'for waterbirds it ships with the dataset.')

        self.root, self.preprocess = root, preprocess
        wanted = SPLIT_CODES[split]
        self.records = []

        with open(metadata, 'r', encoding='utf-8') as f:
            idx = {n: i for i, n in enumerate(f.readline().strip().split(','))}
            for line in f:
                parts = line.strip().split(',')
                if len(parts) < 4 or int(parts[idx['split']]) != wanted:
                    continue
                self.records.append((parts[idx['img_filename']],
                                     int(parts[idx['y']]), int(parts[idx['place']])))

        if limit:
            self.records = self.records[:limit]
        logger.info(f'{name} {split}: {len(self.records)} images')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, i):
        name, y, place = self.records[i]
        image = Image.open(os.path.join(self.root, name)).convert('RGB')
        return self.preprocess(image), torch.tensor(y), torch.tensor(place)


# Back-compat alias: this class was WaterbirdsDataset before it became generic.
WaterbirdsDataset = MetadataDataset


def make_loader(root, preprocess, batch_size, split, limit=None, shuffle=False,
                name='dataset', manifest=None):
    return DataLoader(MetadataDataset(root, preprocess, split, limit, name, manifest),
                      batch_size=batch_size, shuffle=shuffle, num_workers=0)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def normalized_mse(x: torch.Tensor, x_hat: torch.Tensor) -> torch.Tensor:
    return (((x_hat - x) ** 2).mean(dim=-1) / (x ** 2).mean(dim=-1)).mean()


@torch.no_grad()
def unit_norm_decoder(sae: RouteSAE) -> None:
    w = sae.sae.decoder.weight.data
    w /= w.norm(dim=0, keepdim=True)


def warmup_schedule(optimizer, total_steps: int, warmup_frac: float = 0.05):
    warmup = max(1, int(total_steps * warmup_frac))
    decay_start = total_steps - max(1, total_steps // 5)

    def lr_lambda(step):
        if step < warmup:
            return step / warmup
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(1, total_steps // 5))

    return LambdaLR(optimizer, lr_lambda)


def train(args, clip_model, preprocess, device) -> RouteSAE:
    """Train a RouteSAE on CLIP activations and save it."""
    hidden_size, n_layers = clip_dims(clip_model)
    sae = RouteSAE(hidden_size, n_layers, args.latent_size, args.k).to(device)
    sae.train()

    loader = make_loader(args.data, preprocess, args.batch_size, args.train_split,
                         args.limit, shuffle=True, name=getattr(args, 'dataset', 'dataset'),
                         manifest=getattr(args, 'ft_manifest', None))
    optimizer = Adam(sae.parameters(), lr=args.lr, betas=(0.9, 0.999))
    scheduler = warmup_schedule(optimizer, args.epochs * len(loader))
    layer_hist = torch.zeros(sae.n_routed_layers)

    logger.info(f'Training RouteSAE: hidden={hidden_size} layers={n_layers} '
                f'latent={args.latent_size} k={args.k}')

    for epoch in range(args.epochs):
        for step, batch in enumerate(loader):
            pixel_values = batch[0].to(device)
            stack = clip_layer_stack(clip_model, pixel_values, n_layers)
            x, _, _ = pre_process(stack)

            blw, routed, _, x_hat, _ = sae(x, args.aggre, args.routing)
            loss = normalized_mse(routed, x_hat)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()

            layer_hist += blw.sum(dim=(0, 1)).detach().float().cpu()
            if step % args.log_every == 0:
                logger.info(f'  epoch {epoch + 1}/{args.epochs} step {step + 1}/{len(loader)} '
                            f'loss {loss.item():.4f}')
            if step % args.renorm_every == 0:
                unit_norm_decoder(sae)

    unit_norm_decoder(sae)
    os.makedirs(os.path.dirname(args.sae) or '.', exist_ok=True)
    torch.save(sae.state_dict(), args.sae)
    logger.info(f'Saved SAE to {args.sae}')

    total = layer_hist.sum()
    logger.info('Layer routing over training patches:')
    for j, v in enumerate(layer_hist):
        logger.info(f'  layer {sae.start_layer + j:2}: {100 * v / total:5.1f}%')
    return sae


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(args, sae: RouteSAE, clip_model, preprocess, device) -> Dict:
    """NormMSE, zero-shot accuracy original vs reconstructed, group breakdown."""
    _, n_layers = clip_dims(clip_model)
    loader = make_loader(args.data, preprocess, args.batch_size, args.eval_split,
                         args.limit, name=getattr(args, 'dataset', 'dataset'))
    text_embeds = text_class_embeddings(clip_model, device)

    mse_sum = n_batches = 0
    correct_o = correct_r = agree = total = 0
    groups = {g: {'orig': 0, 'recon': 0, 'n': 0} for g in GROUP_NAMES}
    classes = {0: {'orig': 0, 'recon': 0, 'n': 0}, 1: {'orig': 0, 'recon': 0, 'n': 0}}
    seen = torch.zeros(sae.latent_size, dtype=torch.bool, device=device)

    sae.eval()
    for batch in loader:
        pixel_values, y, place = batch[0].to(device), batch[1].to(device), batch[2].to(device)

        stack = clip_layer_stack(clip_model, pixel_values, n_layers)
        x, _, _ = pre_process(stack)
        blw, routed, latents, x_hat, _ = sae(x, args.aggre, args.routing)

        mse_sum += normalized_mse(routed, x_hat).item()
        n_batches += 1
        seen |= (latents != 0).any(dim=tuple(range(latents.dim() - 1)))

        embeds_o = clip_embeds_original(clip_model, pixel_values)
        handles = hook_routesae(sae, clip_model, blw, aggre=args.aggre, routing=args.routing)
        try:
            embeds_r = clip_image_embeds(clip_model, clip_forward_last_hidden(clip_model, pixel_values))
        finally:
            for h in handles:
                h.remove()

        pred_o = (embeds_o @ text_embeds.T).argmax(-1)
        pred_r = (embeds_r @ text_embeds.T).argmax(-1)
        correct_o += (pred_o == y).sum().item()
        correct_r += (pred_r == y).sum().item()
        agree += (pred_o == pred_r).sum().item()
        total += y.numel()

        for i in range(y.numel()):
            for bucket, key in ((groups, (int(y[i]), int(place[i]))), (classes, int(y[i]))):
                bucket[key]['n'] += 1
                bucket[key]['orig'] += int(pred_o[i] == y[i])
                bucket[key]['recon'] += int(pred_r[i] == y[i])

    results = {
        'split': args.eval_split,
        'images': total,
        'normalized_mse': mse_sum / max(1, n_batches),
        'zero_shot_acc_original': correct_o / max(1, total),
        'zero_shot_acc_reconstructed': correct_r / max(1, total),
        'prediction_agreement': agree / max(1, total),
        'live_features': int(seen.sum()),
        'dead_features': int(sae.latent_size - int(seen.sum())),
        'groups': {},
    }
    for key, name in (('orig', 'original'), ('recon', 'reconstructed')):
        recalls = [classes[c][key] / classes[c]['n'] for c in classes if classes[c]['n']]
        results[f'balanced_acc_{name}'] = sum(recalls) / max(1, len(recalls))

    worst_o = worst_r = 1.0
    for g, name in GROUP_NAMES.items():
        n = groups[g]['n']
        if not n:
            continue
        acc_o, acc_r = groups[g]['orig'] / n, groups[g]['recon'] / n
        results['groups'][name] = {'n': n, 'original': acc_o, 'reconstructed': acc_r}
        worst_o, worst_r = min(worst_o, acc_o), min(worst_r, acc_r)
    results['worst_group_original'] = worst_o
    results['worst_group_reconstructed'] = worst_r
    return results


def report(results: Dict) -> None:
    logger.info('=' * 58)
    logger.info(f"Split {results['split']}: {results['images']} images")
    logger.info(f"Normalized MSE          : {results['normalized_mse']:.4f}")
    logger.info(f"Zero-shot (original)    : {results['zero_shot_acc_original']:.4f}")
    logger.info(f"Zero-shot (recon)       : {results['zero_shot_acc_reconstructed']:.4f}")
    logger.info(f"Balanced (original)     : {results['balanced_acc_original']:.4f}")
    logger.info(f"Balanced (recon)        : {results['balanced_acc_reconstructed']:.4f}")
    logger.info(f"Prediction agreement    : {results['prediction_agreement']:.4f}")
    logger.info(f"Worst group orig/recon  : {results['worst_group_original']:.4f} / "
                f"{results['worst_group_reconstructed']:.4f}")
    logger.info(f"Live features           : {results['live_features']} "
                f"(+{results['dead_features']} dead)")
    for name, g in results['groups'].items():
        logger.info(f"  {name:22} n={g['n']:5}  {g['original']:.4f} -> {g['reconstructed']:.4f}")
    logger.info('=' * 58)


def resolve_device(spec: str) -> torch.device:
    if spec != 'auto':
        return torch.device(spec)
    if torch.cuda.is_available():
        return torch.device('cuda:0')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def main() -> None:
    p = argparse.ArgumentParser(description='Train/evaluate RouteSAE on CLIP ViT')
    p.add_argument('--clip', required=True,
                   help="'ViT-B/32', a fine-tuned .pt, or a HuggingFace model id")
    p.add_argument('--dataset', default='waterbirds',
                   help='Dataset name in dataset_settings.REGISTRY (waterbirds, celeba, ...). '
                        'Selects the zero-shot prompts and group names used for evaluation.')
    p.add_argument('--data', required=True,
                   help='Dataset root. Waterbirds: the folder holding metadata.csv. '
                        'CelebA: the parent of celeba/ (the spec appends it).')
    p.add_argument('--sae', default=None, help='Checkpoint path (default derives from --clip)')
    p.add_argument('--out', default='routesae_results', help='Folder for the results JSON')
    p.add_argument('--latent_size', type=int, default=16384)
    p.add_argument('--k', type=int, default=32)
    p.add_argument('--aggre', default='sum', choices=['sum', 'mean'])
    p.add_argument('--routing', default='hard', choices=['hard', 'soft'])
    p.add_argument('--epochs', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=32)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--train_split', default='train', choices=['train', 'val', 'test'])
    p.add_argument('--eval_split', default='test', choices=['train', 'val', 'test'])
    p.add_argument('--ft_manifest', default=None,
                   help="CELEBA ONLY. Train on the exact images listed in a run dir's "
                        "ft_train_manifest.csv (the set its CLIP was fine-tuned on) instead "
                        "of the dataset's train split; --train_split is then ignored. The "
                        "waterbirds SAEs use the full train split and must keep doing so -- "
                        "see _read_ft_manifest for why CelebA differs. Affects training only, "
                        "never evaluation, and does not change the checkpoint name.")
    p.add_argument('--limit', type=int, default=None, help='Cap images (smoke tests)')
    p.add_argument('--device', default='auto')
    p.add_argument('--eval_only', action='store_true', help='Skip training, load --sae')
    p.add_argument('--log_every', type=int, default=20)
    p.add_argument('--renorm_every', type=int, default=10)
    args = p.parse_args()

    spec = _spec_for(args.dataset)
    # metadata.csv lives at the root for waterbirds, under celeba/ for CelebA.
    args.data = spec.metadata_path(args.data)
    logger.info(f'Dataset {spec.name}: prompts={ZERO_SHOT_PROMPTS} | '
                f'{len(GROUP_NAMES)} groups | data={args.data}')

    if args.sae is None:
        stem = os.path.basename(os.path.dirname(args.clip)) if args.clip.endswith(('.pt', '.pth')) \
            else args.clip.replace('/', '~')
        suffix = f'_lim{args.limit}' if args.limit else ''
        # The dataset goes in the checkpoint name for every dataset except
        # waterbirds, whose existing names (routesae_K32_ViT-B~32_16384.pt)
        # predate this and are referenced by the batch scripts and manifests.
        ds_tag = '' if spec.name == 'waterbirds' else f'{spec.name}_'
        args.sae = os.path.join('routesae_weights',
                                f'routesae_K{args.k}_{ds_tag}{stem}_{args.latent_size}{suffix}.pt')

    device = resolve_device(args.device)
    preprocess, clip_model = load_clip(args.clip, device)

    if args.eval_only:
        hidden_size, n_layers = clip_dims(clip_model)
        sae = load_routesae(args.sae, hidden_size, n_layers, args.latent_size, args.k, device)
        logger.info(f'Loaded SAE from {args.sae}')
    else:
        sae = train(args, clip_model, preprocess, device)

    results = evaluate(args, sae, clip_model, preprocess, device)
    report(results)

    results['dataset'] = spec.name
    results['prompts'] = ZERO_SHOT_PROMPTS
    os.makedirs(args.out, exist_ok=True)
    out_path = os.path.join(
        args.out, f'{os.path.splitext(os.path.basename(args.sae))[0]}_{args.eval_split}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'Saved results to {out_path}')


if __name__ == '__main__':
    main()
