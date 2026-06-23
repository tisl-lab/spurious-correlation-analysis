import os
import sys
import glob
import warnings
import torch
import argparse
import logging
import numpy as np
from tqdm import tqdm
from PIL import Image

# Add project root (one level above msae/) so clip_zero_shot is importable
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

import clip
# from datasets import load_dataset
from torch.utils.data import Dataset
from torchvision.datasets import CelebA
from clip_zero_shot import CLIPZeroShot  # noqa: E402

from .utils import get_device, set_seed

"""
CLIP Embedding Extraction Utility

This script extracts embeddings from various datasets using the CLIP model.
It supports multiple datasets and CLIP model variants, and saves the extracted
embeddings as memory-mapped numpy arrays for efficient storage and access.
"""

# Supported CLIP model variants
SUPPORTED_MODELS = [
    "ViT-B~32",
    "ViT-B~16",
    "RN50",
    "ViT-L~14",
]

# Supported image datasets
SUPPORTED_DATASETS = [
    "imagenet",
    "cc3m",
    "celeba",
    "waterbirds",
]

# Supported text vocabulary sources with their file paths
SUPPORTED_VOCABS = {
    "mscoco": "msae/vocab/mscoco_unigram.txt",
    "laion_unigram": "msae/vocab/laion_400_unigram.txt",
    "laion_bigrams": "msae/vocab/laion_400_bigram.txt",
    "laion": ["laion_unigram", "laion_bigrams"],  # Combined vocabulary
    "disect": "msae/vocab/clip_disect_20k.txt",
}

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """
    Parse command line arguments for the embedding extraction script.
    
    Returns:
        argparse.Namespace: Parsed command line arguments
    """
    parser = argparse.ArgumentParser(description="Extract embeddings from the dataset")
    parser.add_argument("-d", "--dataset", type=str, default="waterbirds",
                       help="Dataset to use (one of: imagenet, cc3m, celeba, or a supported vocabulary)")
    parser.add_argument("-m", "--model", type=str, default="ViT-B~32",
                       help="CLIP model variant to use (e.g., 'ViT-B~32' for 'ViT-B/32')")
    parser.add_argument("-f", "--load_ftmodel", action="store_true", default=False,
                       help="Load a fine-tuned CLIP model instead of the pretrained weights")
    parser.add_argument("-b", "--batch-size", type=int, default=4096,
                       help="Batch size for embedding extraction")
    parser.add_argument("-s", "--train-split", default=True, action="store_true",
                       help="Use training split instead of validation/test split")
    parser.add_argument("-v", "--vocab-size", type=int, default=-1,
                       help="Vocabulary size limit (-1 for full vocabulary)")
    parser.add_argument("-w", "--workers", type=int, default=12,
                       help="Number of workers for data loading")
    parser.add_argument("--data_dir", type=str, default="data/waterbirds",
                       help="Root directory for Waterbirds dataset (contains metadata.csv)")
    parser.add_argument("--output_dir", type=str, default="results/waterbirds/embeddings",
                       help="Directory to save extracted embeddings")
    parser.add_argument("--ft_manifest", type=str, default=None,
                       help="Path to ft_train_manifest.csv from a saved run. "
                            "When set, loads the exact fine-tune subset instead of the full train split.")
    # Fine-tuned model selection (used when --load_ftmodel is set)
    parser.add_argument("--ft_run_dir", type=str, default=None,
                       help="Explicit path to a run folder (contains model.pt). "
                            "If not set, the run is matched by --max_samples / --seed / --biased_ft.")
    parser.add_argument("--max_samples", type=int, default=400,
                       help="Max samples per class used during fine-tuning (for run matching)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed used during fine-tuning (for run matching)")
    parser.add_argument("--biased_ft", action="store_true", default=True,
                       help="Whether the fine-tuned run used biased (spurious-aligned) training data")
    return parser.parse_args()


class VocabDataset(Dataset):
    """
    Dataset for processing text vocabulary items.
    
    This dataset treats each vocabulary entry as a text item to be embedded,
    with no corresponding image.
    
    Args:
        data (list): List of vocabulary items (strings)
    """
    def __init__(self, data):
        self.data = data
    
    def __getitem__(self, index):
        """Yield vocabulary items with None as placeholder for images."""
        return None, self.data[index]
    
    def __len__(self):
        """Return the number of vocabulary items."""
        return len(self.data)
    
    def __add__(self, other):
        """Concatenate two VocabDataset instances."""
        if isinstance(other, VocabDataset):
            return VocabDataset(self.data + other.data)
        else:
            raise TypeError("Can only concatenate VocabDataset instances")


class CelebAMy(Dataset):
    """
    Custom wrapper for the CelebA dataset.
    
    Combines multiple attribute labels into a comma-separated string
    to use as the text component for CLIP.
    
    Args:
        root (str): Root directory for CelebA data
        split (str): Dataset split ('train', 'valid', or 'test')
        **kwargs: Additional arguments to pass to CelebA constructor
    """
    def __init__(self, root, split, **kwargs):
        self.celeba = CelebA(root, split=split, **kwargs)
        self.attr_names = self.celeba.attr_names[:40]  # Using first 40 attributes
    
    def __getitem__(self, index):
        """Yield image samples with concatenated attribute labels as text."""
        sample, target = self.celeba[index]
        # Get the indices of attributes that are True for this sample
        labels_by_target = torch.nonzero(target)[:, 0]
        # Convert attribute indices to attribute names and join with commas
        target = ','.join([str(self.attr_names[x]) for x in labels_by_target])
        return sample, target
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return len(self.celeba)


class HFDataset(Dataset):
    """
    Base class for Hugging Face dataset wrappers.
    
    Provides common functionality for loading and preprocessing datasets
    from the Hugging Face hub.
    
    Args:
        dataset (str): Dataset identifier on Hugging Face hub
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use
        download_full (bool): Whether to download the full dataset at once or stream it
        **kwargs: Additional arguments to pass to load_dataset
    """
    def __init__(self, dataset, preprocess, split, download_full=False, **kwargs):
        stream = not download_full
        self.dataset = load_dataset(dataset, split=split, streaming=stream, 
                                    trust_remote_code=True, **kwargs)
        self.preprocess = preprocess
        self.len: int = 0  # Will be set by child classes
    
    def __len__(self):
        """Return the number of samples in the dataset."""
        return self.len
    
    def __getitem__(self, idx):
        """Get a single sample from the dataset."""
        item = self.dataset[idx]
        sample, target = item['jpg'], item['txt']
        if self.preprocess:
            sample = self.preprocess(sample)
        return sample, target


class ImageNetDataset(HFDataset):
    """
    Wrapper for the ImageNet dataset from Hugging Face.
    
    Handles ImageNet-specific data format and preprocessing.
    
    Args:
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use ('train' or 'validation')
    """
    def __init__(self, preprocess, split):
        super().__init__("ILSVRC/imagenet-1k", preprocess, split, True)
        # Set dataset length based on split info
        self.len = self.dataset.info.splits[self.dataset.split].num_examples
        # Create mapping from class names to indices
        self.class_to_idx = {}
        for idx, class_name in enumerate(self.dataset.info.features['label'].names):
            self.class_to_idx[class_name] = idx
    
    def __getitem__(self, idx):
        try:
            item = self.dataset[idx]
            sample, target = item['image'], item['label']
            if target == -1:
                target = ""  # Handle missing labels
            else:
                target = self.dataset.info.features['label'].int2str(target)
            
            if isinstance(sample, Image.Image):
                sample = sample.convert("RGB")
            if self.preprocess:
                sample = self.preprocess(sample)
            return sample, target
        except (UnicodeDecodeError, OSError, SyntaxError) as e:
            return None, None

class CC3MDataset(HFDataset):
    """
    Wrapper for the Conceptual Captions 3M dataset.
    
    Handles CC3M-specific data format and preprocessing.
    
    Args:
        preprocess (callable): Image preprocessing function
        split (str): Dataset split to use ('train' or 'validation')
        download_full (bool): Whether to download the full dataset at once
    """
    def __init__(self, preprocess, split, download_full=False):
        download_full=True
        super().__init__("pixparse/cc3m-wds", preprocess, split, download_full)
        # Hardcoded dataset sizes since they're not always available from the API
        if split == "train":
            self.len = 2905954
        else:
            self.len = 13443
    
    def __getitem__(self, idx):
        """Return preprocessed image and caption pairs."""
        try:
            item = self.dataset[idx]
            sample, target = item['jpg'], item['txt']
            if isinstance(sample, Image.Image):
                sample = sample.convert("RGB")
            
            if self.preprocess:
                sample = self.preprocess(sample)
            return sample, target
        except (UnicodeDecodeError, OSError, SyntaxError) as e:
            return None, None 


class EmbeddingExtractor:
    """
    Utility for extracting CLIP embeddings from images and text.
    
    Handles loading the CLIP model and preprocessing pipeline, and provides
    methods for embedding both images and text.
    
    Args:
        model_name (str): Name of the CLIP model variant to use
        device (str): Device to run inference on ('cuda' or 'cpu')
    """
    def __init__(self, model_name, device, ft_model=False, model_path=None) -> None:
        self.model_name = model_name
        self.device = device
        # Load the model, preprocessor, and tokenizer
        self.model, self.preprocessor, self.tokenizer, self.token_max_length = self.load_model(
            model_name, self.device, ft_model=ft_model, model_path=model_path)

        self.model = self.model.to(self.device)
        self.model.eval()
    
    @staticmethod
    def load_model(model_name, device, ft_model = False ,model_path=None):
        """
        Load a CLIP model, preprocessor, and tokenizer.
        
        Args:
            model_name (str): Name of the CLIP model variant
            device (str): Device to load the model on
            model_path (str, optional): Custom path for model weights
            
        Returns:
            tuple: (model, preprocessor, tokenizer, token_max_length)
            
        Raises:
            ValueError: If the model variant is not supported
        """
        if model_name not in SUPPORTED_MODELS:
            raise ValueError(f"Model {model_name} not supported, please use one of {SUPPORTED_MODELS}")
        
        # Replace '~' with '/' in model name (to handle command line limitations)
        model_name = model_name.replace("~", "/")
        
        # Load the model and preprocessor
        if not ft_model:
            model, preprocessor = clip.load(model_name, device=device, download_root=model_path)
        else:
            if model_path is None:
                raise ValueError("model_path must be provided when ft_model=True")
            clip_ft = CLIPZeroShot.load_model(model_path, device=device)
            model, preprocessor = clip_ft.model, clip_ft.preprocess
        
        # Create a tokenizer that truncates by default
        original_tokenizer = clip.tokenize
        tokenizer = lambda x: original_tokenizer(x, truncate=True)
        token_max_length = 77  # Standard context length for CLIP
            
        return model, preprocessor, tokenizer, token_max_length
    
    def embed_text(self, text):
        """
        Extract text embeddings using the CLIP model.
        
        Args:
            text (list or str or torch.Tensor): Text input(s) to embed
            
        Returns:
            tuple: (text_features, tokenized_text)
                - text_features: Normalized text embeddings
                - tokenized_text: Tokenized text inputs
        """
        # Handle different input types
        if not isinstance(text, torch.Tensor):
            text_embeddings = self.tokenizer(text if isinstance(text, list) else [text]).to(self.device)
        else:
            text_embeddings = text.to(self.device)

        # Extract features with mixed precision
        with torch.no_grad(), torch.amp.autocast("cuda"):
            text_features = self.model.encode_text(text_embeddings)
            
        return text_features, text_embeddings.detach().cpu()
    
    def embed_image(self, img):
        """
        Extract image embeddings using the CLIP model.
        
        Args:
            img (list or PIL.Image or torch.Tensor): Image input(s) to embed
            
        Returns:
            tuple: (image_features, preprocessed_images)
                - image_features: Normalized image embeddings
                - preprocessed_images: Preprocessed image tensors
        """
        # Handle different input types
        if isinstance(img, list):
            if not isinstance(img[0], torch.Tensor):
                img = [self.preprocessor(i).to(self.device) for i in img]
            img = torch.stack(img, dim=0).to(self.device)
        else:
            if not isinstance(img, torch.Tensor):
                img = self.preprocessor(img)
            
        if img.ndim == 3:
            img = img.unsqueeze(0)
        
        try:
            img = img.to(self.device)
        except Exception as e:
            pass

        # Extract features with mixed precision
        with torch.no_grad(), torch.amp.autocast("cuda"):
            image_features = self.model.encode_image(img)
            
        return image_features, img.detach().cpu()
    

class WaterbirdsDataset(Dataset):
    """
    Waterbirds dataset for CLIP embedding extraction.

    __getitem__ returns (preprocessed_image, class_name) to stay compatible
    with the (image, text) convention used by EmbeddingExtractor.
    Call get_metadata() to retrieve a DataFrame with label / group_id / bg
    for every sample.

    Split codes in metadata.csv: 0=train, 1=val, 2=test.
    """

    CLASS_NAMES = ["landbird", "waterbird"]
    BG_NAMES    = ["land", "water"]
    GROUP_NAMES = [
        "landbird / land bg",
        "landbird / water bg",
        "waterbird / land bg",
        "waterbird / water bg",
    ]

    def __init__(self, root: str, preprocess, split=None, ft_manifest: str = None):
        import pandas as pd

        self.preprocess = preprocess
        self.samples = []

        if ft_manifest is not None:
            # Load exact fine-tune set from saved manifest (run_waterbirds_msae.py output)
            if not os.path.isfile(ft_manifest):
                raise FileNotFoundError(f"ft_manifest not found: {ft_manifest}")
            df = pd.read_csv(ft_manifest)
            for _, row in df.iterrows():
                self.samples.append({
                    "path":     row["img_path"],
                    "label":    int(row["label"]),
                    "class":    row["class"],
                    "bg":       row["bg"],
                    "group_id": int(row["group_id"]),
                    "group":    row["group"],
                    "aligned":  bool(row["aligned"]),
                })
        else:
            meta_path = os.path.join(root, "metadata.csv")
            if not os.path.isfile(meta_path):
                raise FileNotFoundError(
                    f"Waterbirds metadata not found: {meta_path}\n"
                    "Download from: https://nlp.stanford.edu/data/dro/"
                    "waterbird_complete95_forest2water2.tar.gz"
                )
            df = pd.read_csv(meta_path)
            if split is not None:
                splits = [split] if isinstance(split, int) else list(split)
                df = df[df["split"].isin(splits)].reset_index(drop=True)
            for _, row in df.iterrows():
                label = int(row["y"])
                place = int(row["place"])
                gid   = label * 2 + place
                self.samples.append({
                    "path":     os.path.join(root, row["img_filename"]),
                    "label":    label,
                    "class":    self.CLASS_NAMES[label],
                    "bg":       self.BG_NAMES[place],
                    "group_id": gid,
                    "group":    self.GROUP_NAMES[gid],
                    "aligned":  gid in {0, 3},
                })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s   = self.samples[idx]
        img = Image.open(s["path"]).convert("RGB")
        if self.preprocess:
            img = self.preprocess(img)
        return img, s["class"]

    def get_metadata(self):
        import pandas as pd
        return pd.DataFrame(self.samples)


def resolve_ft_run(args):
    """
    Locate the saved fine-tuned CLIP run folder and return (run_dir, model_pt_path).

    If args.ft_run_dir is given it is used directly.
    Otherwise, all clip_ft_*/model.pt files under results/<dataset>/ are scanned
    and the first one matching ft_tag / max_samples / seed is returned.
    """
    if args.ft_run_dir:
        model_pt = os.path.join(args.ft_run_dir, "model.pt")
        if not os.path.isfile(model_pt):
            raise FileNotFoundError(f"model.pt not found in --ft_run_dir: {args.ft_run_dir}")
        return args.ft_run_dir, model_pt

    ft_tag = "BIASED" if args.biased_ft else "FULL"
    # Run folders live one level above the embeddings output dir
    runs_base = os.path.dirname(args.output_dir)
    candidates = glob.glob(os.path.join(runs_base, "clip_ft_*", "model.pt"))
    if not candidates:
        raise FileNotFoundError(
            f"No fine-tuned run folders found under {runs_base}. "
            "Run run_waterbirds_msae.py first, or pass --ft_run_dir explicitly."
        )
    for model_pt in sorted(candidates):
        payload = torch.load(model_pt, map_location="cpu", weights_only=False)
        meta = payload.get("metadata", {})
        if (meta.get("ft_tag") == ft_tag
                and meta.get("max_samples") == args.max_samples
                and meta.get("seed") == args.seed):
            return os.path.dirname(model_pt), model_pt
    raise FileNotFoundError(
        f"No run matched ft_tag={ft_tag}, max_samples={args.max_samples}, seed={args.seed} "
        f"under {runs_base}."
    )


def load_data(dataset, preprocess, train=False, data_dir=None, ft_manifest=None):
    """
    Load a dataset by name with appropriate preprocessing.

    Args:
        dataset (str): Name of the dataset to load
        preprocess (callable): Image preprocessing function
        train (bool): Whether to load training split (True) or validation/test split (False)
        ft_manifest (str, optional): Path to ft_train_manifest.csv to load the exact
            fine-tune subset instead of the full training split (Waterbirds only).

    Returns:
        IterableDataset: The loaded dataset

    Raises:
        ValueError: If the dataset is not supported
    """
    if dataset == "imagenet":
        dataset = ImageNetDataset(preprocess, "train" if train else "validation")
    elif dataset == "cc3m":
        dataset = CC3MDataset(preprocess, "train" if train else "validation")
    elif dataset == "celeba":
        dataset = CelebAMy(download=True, split="train" if train else "test",
                           transform=preprocess, target_type="attr")
    elif dataset == "waterbirds":
        if ft_manifest is not None:
            dataset = WaterbirdsDataset(data_dir or "", preprocess, ft_manifest=ft_manifest)
        else:
            if data_dir is None:
                raise ValueError(
                    "Waterbirds requires --data_dir pointing to the dataset root "
                    "(directory containing metadata.csv)."
                )
            split_code = 0 if train else 2   # 0=train, 2=test
            dataset = WaterbirdsDataset(data_dir, preprocess, split=split_code)
    else:
        raise ValueError(f"Dataset {dataset} not supported, please use one of {SUPPORTED_DATASETS}")

    # Add reverse class mapping if available
    if hasattr(dataset, "class_to_idx"):
        dataset.idx_to_class = {v: k for k, v in dataset.class_to_idx.items()}
    
    return dataset


def load_vocab(vocab, vocab_size=-1):
    """
    Load a text vocabulary from files.
    
    Args:
        vocab (str): Name of the vocabulary to load
        vocab_size (int): Maximum number of items to include (-1 for all)
        
    Returns:
        VocabDataset: Dataset containing the vocabulary items
        
    Raises:
        ValueError: If the vocabulary is not supported
    """
    if vocab not in SUPPORTED_VOCABS.keys():
        raise ValueError(f"Vocab {vocab} not supported, please use one of {SUPPORTED_VOCABS.keys()}")
    
    path = SUPPORTED_VOCABS[vocab]
    
    # Handle composite vocabularies (e.g., "laion" = unigrams + bigrams)
    if isinstance(path, list):
        current_vocab = None
        for x in path:
            if current_vocab is None:
                current_vocab = load_vocab(x, vocab_size // 2 if vocab_size > 0 else -1)
            else:
                current_vocab += load_vocab(x, vocab_size // 2 if vocab_size > 0 else -1)
        return current_vocab

    # Load vocabulary from file
    vocab_data = []
    with open(path, 'r') as f:
        lines = f.readlines()
        if vocab_size > 0:
            # Apply vocabulary size limit if specified
            if vocab_size > len(lines):
                warnings.warn(f"Vocab size {vocab_size} is greater than the actual vocab size {len(lines)}. Using full vocab.")
            else:
                lines = lines[-vocab_size:]  # Take most frequent terms (assuming frequency-sorted lists)

        for line in lines:
            line = line.strip()
            vocab_data.append(line)
    
    return VocabDataset(vocab_data)


def safe_collate(batch):
    # Filter out None values
    batch = [(img, txt) for img, txt in batch if img is not None or txt is not None]
    if not batch:
        return None, None  # Handle empty batch case
    
    # If any item is None, set it to None
    images, texts = zip(*batch)
    if any(x is None for x in images):
        images = None
    else:
        images = torch.stack(images)
        
    if any(x is None for x in texts):
        texts = None
        
    # Process valid items
    return images, texts


def main(args):
    """
    Main function for extracting and saving embeddings.
    
    Loads the specified dataset and model, extracts embeddings for all samples,
    and saves them to disk as memory-mapped arrays.
    
    Args:
        args (argparse.Namespace): Command line arguments
    """
    set_seed(42)
    logger.info(f"Extracting embeddings for {args.dataset} using {args.model} model")
    device = get_device()
    logger.info(f"Using device: {device}")
    
    # Load CLIP model and embedding extractor
    if args.load_ftmodel:
        run_dir, model_pt = resolve_ft_run(args)
        logger.info(f"Loading fine-tuned model from {model_pt}")
        extractor = EmbeddingExtractor(args.model, device, ft_model=True, model_path=model_pt)
        ft_suffix = f"_ft{args.max_samples}"
        logger.info(f"Fine-tuned model loaded (run: {run_dir})")
    else:
        extractor = EmbeddingExtractor(args.model, device)
        ft_suffix = ""

    os.makedirs(args.output_dir, exist_ok=True)

    # Load appropriate dataset
    if args.dataset in SUPPORTED_DATASETS:
        logger.info(f"Loading dataset {args.dataset}")
        dataset = load_data(args.dataset, extractor.preprocessor,
                            args.train_split, data_dir=args.data_dir,
                            ft_manifest=args.ft_manifest)
        if args.ft_manifest:
            split_name = "fttrain"
        else:
            split_name = "train" if args.train_split else "test"

        # Save Waterbirds metadata (label / group_id / bg) alongside embeddings
        if args.dataset == "waterbirds":
            meta_path = os.path.join(
                args.output_dir,
                f"waterbirds_{args.model}_{split_name}_metadata.csv",
            )
            dataset.get_metadata().to_csv(meta_path, index=False)
            logger.info(f"Waterbirds metadata saved to {meta_path}")
    elif args.dataset in SUPPORTED_VOCABS.keys():
        logger.info(f"Loading vocab {args.dataset}")
        dataset = load_vocab(args.dataset, args.vocab_size)
        split_name = str(args.vocab_size)
    else:
        raise ValueError(f"Dataset {args.dataset} not supported, please use one of {SUPPORTED_DATASETS + list(SUPPORTED_VOCABS.keys())}")
    
    with torch.no_grad(), torch.amp.autocast("cuda"):
        # Get a sample to determine embedding dimension
        dataset_size = len(dataset)
        sample = dataset[0][0]
        if sample is None:
            sample = dataset[0][1]
            if sample is None:
                logger.error("Sample is None, skipping embedding extraction.")
                return
            features, _ = extractor.embed_text(sample)
        else:
            features, _ = extractor.embed_image(sample)
        embedding_dim = features.shape[-1]
        
        logger.info(f"Creating embeddings Memmap with length {dataset_size} and shape {embedding_dim}")
        
        # Prepare memory-mapped arrays for storing embeddings
        memmap_image_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_image_{dataset_size}_{embedding_dim}.npy"
        )
        image_memmap = np.memmap(
            memmap_image_path,
            dtype=np.float32,
            mode='w+',
            shape=(dataset_size, embedding_dim)
        )
        logger.info(f"Saving image embeddings to {memmap_image_path}")

        memmap_text_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_text_{dataset_size}_{embedding_dim}.npy"
        )
        text_memmap = np.memmap(
            memmap_text_path, 
            dtype=np.float32, 
            mode='w+', 
            shape=(dataset_size, embedding_dim)
        )
        logger.info(f"Saving text embeddings to {memmap_text_path}")
        
        # Also save the original text for reference
        text_output_path = os.path.join(
            args.output_dir,
            f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_text_{dataset_size}.txt"
        )
        logger.info(f"Saving original text to {text_output_path}")
        
        # Process data in batches
        text_full = []
        dl = torch.utils.data.DataLoader(
            dataset, 
            batch_size=args.batch_size, 
            num_workers=args.workers,
            pin_memory=True, 
            drop_last=False,
            collate_fn=safe_collate
        )
        
        logger.info("Extracting embeddings...")
        start_idx = 0
        successful_count = 0
        for images, texts in tqdm(dl, total=len(dl), desc="Extracting embeddings"):
            if images is None and texts is None:
                continue
            
            # Calculate batch indices
            end_idx = start_idx + len(images if images is not None else texts)
            
            # Extract and save image embeddings
            if images is not None:
                image_embeddings, _ = extractor.embed_image(images)
                image_memmap[start_idx:end_idx] = image_embeddings.detach().cpu().numpy().astype(np.float32)
                image_memmap.flush()
            
            # Extract and save text embeddings
            if texts is not None:
                
                # Convert numerical class indices to text labels if needed
                texts = list(texts)
                if isinstance(texts, list) and isinstance(texts[0], int):
                    texts = [dataset.idx_to_class[x] for x in texts]
                
                text_embeddings, _ = extractor.embed_text(texts)
                text_memmap[start_idx:end_idx] = text_embeddings.detach().cpu().numpy().astype(np.float32)
                text_memmap.flush()
            
                # Collect original text
                text_full.extend(texts)
            
            # Update indices for next batch
            start_idx = end_idx
            successful_count += len(images if images is not None else texts)
        
        # Save original text to file
        with open(text_output_path, 'w') as f:
            f.write("\n".join(text_full))
        
        #Correct end index for memmap if needed
        if successful_count < dataset_size:
            logger.info(f"Resizing memmaps to {successful_count} items")
            
            # Create new memmaps with correct size
            final_image_path = os.path.join(
                args.output_dir,
                f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_image_{successful_count}_{embedding_dim}.npy"
            )
            final_text_path = os.path.join(
                args.output_dir,
                f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_text_{successful_count}_{embedding_dim}.npy"
            )
            
            # Copy data to new memmaps
            final_image_memmap = np.memmap(
                final_image_path, 
                dtype=np.float32, 
                mode='w+', 
                shape=(successful_count, embedding_dim)
            )
            final_image_memmap[:] = image_memmap[:successful_count]
            final_image_memmap.flush()
            
            final_text_memmap = np.memmap(
                final_text_path, 
                dtype=np.float32, 
                mode='w+', 
                shape=(successful_count, embedding_dim)
            )
            final_text_memmap[:] = text_memmap[:successful_count]
            final_text_memmap.flush()
            
            # Also update the text file
            final_text_output_path = os.path.join(
                args.output_dir,
                f"{args.dataset}_{args.model}{ft_suffix}_{split_name}_text_{successful_count}.txt"
            )
            with open(final_text_output_path, 'w') as f:
                f.write("\n".join(text_full))
            
            logger.info(f"Resized memmaps saved at {final_image_path} and {final_text_path} and text to {final_text_output_path}")
                        
            # Cleanup original files
            os.remove(memmap_image_path)
            os.remove(memmap_text_path)
            os.remove(text_output_path)
            logger.info(f"Removed original memmaps and text file from {memmap_image_path}, {memmap_text_path}, and {text_output_path}")
            
            memmap_image_path = final_image_path
            image_memmap = final_image_memmap
            memmap_text_path = final_text_path
            text_memmap = final_text_memmap
            
        if image_memmap.sum() == 0:
            os.remove(memmap_image_path)
            logger.info(f"Removed empty memmap file {memmap_image_path}")
        
        if text_memmap.sum() == 0:
            os.remove(memmap_text_path)
            logger.info(f"Removed empty memmap file {memmap_text_path}")
        
        logger.info("Embedding extraction complete")
        
        
if __name__ == "__main__":
    args = parse_args()
    main(args)
