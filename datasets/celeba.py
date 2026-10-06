"""
datasets/celeba.py
==================
CelebA loader for the spurious-correlation pipeline.

The standard benchmark attributes (Sagawa et al., 2020): the target is
Blond_Hair and the spurious attribute is Male. The pipeline fine-tunes on
blond women + not-blond men only (run_celeba_finetune.py), so the learned
shortcut is two-way -- female ⇒ blond, male ⇒ not blond -- and the two
misaligned groups are not-blond women and blond men; the latter is also the
rarest group (~1.4k of 163k training images).

Group encoding, identical to waterbirds' (label * 2 + attribute):

    group_id   label            attribute     role
    0          not blond (0)    female (0)    misaligned
    1          not blond (0)    male   (1)    aligned
    2          blond     (1)    female (0)    aligned
    3          blond     (1)    male   (1)    misaligned (rare)

Expected layout on disk (the official release, or the Kaggle mirror):

    <root>/celeba/
        img_align_celeba/*.jpg
        list_attr_celeba.txt       (official)  or  list_attr_celeba.csv      (Kaggle)
        list_eval_partition.txt    (official)  or  list_eval_partition.csv   (Kaggle)

Both formats are read. The first time a root is opened, a waterbirds-style
`metadata.csv` (img_filename, y, split, place, plus the two attribute names)
is written next to the images, so every later stage of the pipeline sees the
same table shape it sees for waterbirds; after that the loader reads it
directly.

The class mirrors run_waterbirds_msae.WaterbirdsDataset's interface exactly
-- constructor semantics, `samples` tuples, `__getitem__` -- because that is
what the fine-tuning / run-dir producer consumes.

USAGE:
    from datasets import CelebA
    ds = CelebA(root="./data", split=0)             # train
    image, label, attr_name, group_id = ds[0]
    # image     : PIL.Image (RGB), or the CLIP tensor once clip_preprocess is set
    # label     : int, 0 = not blond, 1 = blond
    # attr_name : "female" | "male"
    # group_id  : label * 2 + male
"""

import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["not blond", "blond"]     # index = label  (y in metadata)
ATTR_NAMES  = ["female", "male"]         # index = spurious attribute value (place in metadata)
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}

# group_id = label * 2 + attribute  (same encoding as run_waterbirds_msae)
GROUP_NAMES = [
    "not blond / female",   # group 0 — misaligned
    "not blond / male",     # group 1 — aligned
    "blond / female",       # group 2 — aligned
    "blond / male",         # group 3 — misaligned (also the rarest: ~1.4k of 163k train)
]

# Which groups follow the bias the model is FINE-TUNED with. The biased
# fine-tuning set (run_celeba_finetune.py) is blond women + not-blond men
# only, so the learned shortcut runs both ways -- female ⇒ blond, male ⇒
# not blond -- exactly as waterbirds' land ⇒ landbird / water ⇒ waterbird.
# Two aligned and two misaligned groups, so every group-based method in the
# pipeline sees the same structure it sees on waterbirds.
# (CelebA's natural training distribution biases only one way, blond ⇒
# female; under that reading only group 3 would be misaligned. The
# definitions here follow the fine-tuning design, which is what the
# pipeline's model actually learned.)
ALIGNED_GROUPS    = {1, 2}
MISALIGNED_GROUPS = {0, 3}

# For each misaligned group, the aligned group of the SAME class -- the
# contrast the label-guided finder compares against ("same class, the
# attribute it was correlated with"). Waterbirds: {1: 0, 2: 3}.
CONTRAST_GROUP = {0: 1,          # not blond / female  ->  not blond / male
                  3: 2}          # blond / male        ->  blond / female


def _gid(label: int, attr: int) -> int:
    return label * 2 + attr


def is_aligned(gid: int) -> bool:
    return gid in ALIGNED_GROUPS


def save_dataset_manifest(dataset, path, split_name):
    """Write the manifest the pipeline reads, with exactly the columns
    run_waterbirds_msae.save_dataset_manifest writes:
    img_path, label, class, bg, group_id, group, aligned, split.
    (`bg` keeps its waterbirds name and holds the attribute name here.)"""
    import pandas as pd
    rows = []
    for img_path, label, attr_name, gid in dataset.samples:
        rows.append({
            "img_path": img_path,
            "label":    label,
            "class":    CLASS_NAMES[label],
            "bg":       attr_name,
            "group_id": gid,
            "group":    GROUP_NAMES[gid],
            "aligned":  is_aligned(gid),
            "split":    split_name,
        })
    pd.DataFrame(rows).to_csv(path, index=False)


def _read_attr_table(root):
    """list_attr_celeba.{txt,csv} -> DataFrame indexed by image filename, 0/1 values."""
    import pandas as pd
    txt, csv_ = os.path.join(root, "list_attr_celeba.txt"), os.path.join(root, "list_attr_celeba.csv")
    if os.path.isfile(csv_):
        df = pd.read_csv(csv_)
        df = df.set_index(df.columns[0])
    elif os.path.isfile(txt):
        # Official format: line 1 = image count, line 2 = attribute names,
        # then "<filename> <±1> ... <±1>" rows separated by runs of spaces.
        df = pd.read_csv(txt, sep=r"\s+", skiprows=1, index_col=0)
    else:
        raise FileNotFoundError(
            f"No list_attr_celeba.txt / .csv in {root}\n"
            f"Download CelebA (aligned) from https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html\n"
            f"or the Kaggle mirror, and place img_align_celeba/ plus the two list files under {root}/")
    df.index.name = "img_filename"
    return ((df + 1) // 2).astype(int)          # ±1 -> 0/1


def _read_partition(root):
    """list_eval_partition.{txt,csv} -> Series filename -> split (0/1/2)."""
    import pandas as pd
    txt, csv_ = os.path.join(root, "list_eval_partition.txt"), os.path.join(root, "list_eval_partition.csv")
    if os.path.isfile(csv_):
        df = pd.read_csv(csv_)
        return df.set_index(df.columns[0]).iloc[:, 0].astype(int)
    if os.path.isfile(txt):
        df = pd.read_csv(txt, sep=r"\s+", header=None, index_col=0)
        return df.iloc[:, 0].astype(int)
    raise FileNotFoundError(f"No list_eval_partition.txt / .csv in {root}")


def build_metadata(root, target_attr="Blond_Hair", spurious_attr="Male", force=False):
    """Write <root>/metadata.csv (waterbirds-style) from the CelebA list files
    and return it as a DataFrame. Cached: rebuilt only when missing, when
    `force`, or when it was built for a different attribute pair.

    Columns: img_filename, y, split, place, target_attr, spurious_attr --
    the first four are exactly waterbirds' metadata.csv, so the rest of the
    pipeline can treat the two datasets alike; the last two record which
    CelebA attributes y and place stand for.
    """
    import pandas as pd
    path = os.path.join(root, "metadata.csv")
    if os.path.isfile(path) and not force:
        df = pd.read_csv(path)
        if (df.get("target_attr", pd.Series([target_attr])).iloc[0] == target_attr
                and df.get("spurious_attr", pd.Series([spurious_attr])).iloc[0] == spurious_attr):
            return df
    attrs = _read_attr_table(root)
    for a in (target_attr, spurious_attr):
        if a not in attrs.columns:
            raise KeyError(f"CelebA has no attribute {a!r}; available: {list(attrs.columns)}")
    part = _read_partition(root)
    df = pd.DataFrame({
        "img_filename": [os.path.join("img_align_celeba", f) for f in attrs.index],
        "y":     attrs[target_attr].values,
        "split": part.reindex(attrs.index).values,
        "place": attrs[spurious_attr].values,
    })
    df["target_attr"], df["spurious_attr"] = target_attr, spurious_attr
    df.to_csv(path, index=False)
    return df


class CelebA:
    """
    CelebA wrapper compatible with CLIPZeroShot.run() and fine_tune(), with
    the same interface as run_waterbirds_msae.WaterbirdsDataset.

    Each sample returns:
        image      (PIL.Image or tensor) — face image
        label      (int)                 — 0 = not blond, 1 = blond
        attr_name  (str)                 — "female" or "male"
        group_id   (int)                 — label * 2 + male

    clip_preprocess is injected by CLIPZeroShot automatically.
    Split codes: 0 = train, 1 = val, 2 = test.
    """

    clip_preprocess = None

    def __init__(self, root="./data", split=None, group_filter=None,
                 max_samples=None, seed=42, balanced=False,
                 target_attr="Blond_Hair", spurious_attr="Male"):
        """
        Args:
            root          : parent directory containing 'celeba/'
            split         : None (all), int, or list[int] — 0=train, 1=val, 2=test
            group_filter  : None or set/list of group_ids to keep
            max_samples   : cap per class (or per group when balanced=True)
            seed          : RNG seed for subsampling
            balanced      : subsample max_samples per group (class × attribute)
            target_attr   : CelebA attribute used as the label   (default Blond_Hair)
            spurious_attr : CelebA attribute used as the group   (default Male)
        """
        dataset_root = os.path.join(root, "celeba")
        if not os.path.isdir(os.path.join(dataset_root, "img_align_celeba")):
            raise FileNotFoundError(
                f"CelebA not found at {dataset_root}/img_align_celeba\n"
                f"Expected layout: <root>/celeba/img_align_celeba/*.jpg + list_attr_celeba + list_eval_partition")

        df = build_metadata(dataset_root, target_attr, spurious_attr)

        if split is not None:
            splits = [split] if isinstance(split, int) else list(split)
            df = df[df["split"].isin(splits)].reset_index(drop=True)

        # Group filter BEFORE subsampling (waterbirds does it after), so
        # max_samples per class is drawn from the groups that are kept.
        # CelebA's gender skew makes the order matter: 400 blond images are
        # ~376 women + 24 men, 400 not-blond are ~207 women + 193 men, so
        # filtering after sampling would leave 376 vs 193 -- not a balanced
        # biased set.
        keep = set(group_filter) if group_filter is not None else None
        if keep is not None:
            df = df[(df["y"] * 2 + df["place"]).isin(keep)].reset_index(drop=True)

        rng = np.random.default_rng(seed)
        if max_samples is not None:
            group_cols = ["y", "place"] if balanced else ["y"]
            parts = []
            for _, grp in df.groupby(group_cols):
                if len(grp) > max_samples:
                    chosen = rng.choice(len(grp), size=max_samples, replace=False)
                    parts.append(grp.iloc[sorted(chosen)])
                else:
                    parts.append(grp)
            import pandas as pd
            df = pd.concat(parts).sort_index().reset_index(drop=True)

        self.root = dataset_root
        self.target_attr, self.spurious_attr = target_attr, spurious_attr
        self.samples = []
        for fname, y, place in zip(df["img_filename"], df["y"], df["place"]):
            gid = _gid(int(y), int(place))
            self.samples.append((os.path.join(dataset_root, fname), int(y), ATTR_NAMES[int(place)], gid))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label, attr_name, gid = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.clip_preprocess is not None:
            img = self.clip_preprocess(img)
        return img, label, attr_name, gid

    def get_class_name(self, label: int) -> str:
        return CLASS_NAMES[label]

    def group_counts(self) -> dict:
        counts = {}
        for _, _, _, gid in self.samples:
            counts[gid] = counts.get(gid, 0) + 1
        return dict(sorted(counts.items()))

    def __repr__(self):
        return (f"CelebA(n_samples={len(self)}, target={self.target_attr}, "
                f"spurious={self.spurious_attr}, groups={self.group_counts()})")
