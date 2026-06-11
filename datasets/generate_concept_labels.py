"""
datasets/generate_concept_labels.py
====================================

Two steps:

1. Reads all MiniImageNet_*_Concepts.json files, collects every unique synset
   folder name that appears in them, and writes:

       datasets/concept_class_labels.json
           {"n01532829": "house_finch", ...}

2. For each MiniImageNet_*_Concepts.json, produces a labeled variant where
   every image path has its synset folder replaced with the human-readable
   class name.  Original files are left untouched.

       MiniImageNet_Bird_Concepts.json    →  MiniImageNet_Bird_Concepts_labeled.json
       MiniImageNet_Canidea_Concepts.json →  MiniImageNet_Canidea_Concepts_labeled.json
       MiniImageNet_Insect_Concepts.json  →  MiniImageNet_Insect_Concepts_labeled.json

   Image path format after labeling:
       "n01532829/n0153282900000155.jpg"  →  "house_finch/n0153282900000155.jpg"

USAGE:
    python datasets/generate_concept_labels.py
"""

import glob
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))


def collect_synsets_from_concepts():
    synsets = set()
    for path in sorted(glob.glob(os.path.join(_HERE, "MiniImageNet_*_Concepts.json"))):
        with open(path) as f:
            concept_map = json.load(f)
        for img_list in concept_map.values():
            for rel_path in img_list:
                synsets.add(rel_path.split("/")[0])
    return synsets


def write_concept_class_labels(synset_to_name):
    synsets = collect_synsets_from_concepts()
    print(f"Unique synsets found in concept files: {len(synsets)}")

    result = {}
    missing = []
    for synset in sorted(synsets):
        name = synset_to_name.get(synset)
        if name:
            result[synset] = name
        else:
            missing.append(synset)

    if missing:
        print(f"Warning: no label found for {len(missing)} synset(s): {missing}")

    out_path = os.path.join(_HERE, "concept_class_labels.json")
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Written {len(result)} entries → {out_path}")
    return result


def write_labeled_concept_files(synset_to_name):
    concept_files = sorted(glob.glob(os.path.join(_HERE, "MiniImageNet_*_Concepts.json")))
    for src_path in concept_files:
        with open(src_path) as f:
            concept_map = json.load(f)

        labeled = {}
        for concept, img_list in concept_map.items():
            labeled_imgs = []
            for rel_path in img_list:
                synset, filename = rel_path.split("/", 1)
                label = synset_to_name.get(synset, synset)
                labeled_imgs.append(f"{label}/{filename}")
            labeled[concept] = labeled_imgs

        base = os.path.basename(src_path)                          # MiniImageNet_Bird_Concepts.json
        name_stem = base.replace(".json", "")                      # MiniImageNet_Bird_Concepts
        out_path = os.path.join(_HERE, f"{name_stem}_labeled.json")
        with open(out_path, "w") as f:
            json.dump(labeled, f, indent=2)
        print(f"Written → {out_path}")


def main():
    with open(os.path.join(_HERE, "mini_imagenet_labels.json")) as f:
        synset_to_name = json.load(f)

    print("── Step 1: concept_class_labels.json ──────────────────")
    write_concept_class_labels(synset_to_name)

    print("\n── Step 2: labeled concept files ──────────────────────")
    write_labeled_concept_files(synset_to_name)


if __name__ == "__main__":
    main()
