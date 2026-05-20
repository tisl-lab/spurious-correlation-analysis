import os
import logging
import numpy
from glob import glob
from typing import Any, List, Sequence, Tuple
from pathlib import Path
import pickle
from tqdm import tqdm

import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder



# Mapping the 100 classes of MiniImageNet to the full 1000 classes
MINI_IMAGENET_CLS_TO_IMAGENET_CLS = [12, 15, 51, 64, 70, 96, 99, 107, 111, 121, 149, 166, 173, 176, 207, 214, 228, 242,
                                     244, 245, 249, 251, 256, 266, 270, 275, 279, 291, 299, 301, 306, 310, 359, 364,
                                     392, 403, 412, 427, 440, 454, 471, 476, 478, 484, 494, 502, 503, 507, 519, 524,
                                     533, 538, 546, 553, 556, 567, 569, 584, 597, 602, 604, 605, 629, 655, 657, 659,
                                     683, 687, 702, 709, 713, 735, 741, 758, 779, 781, 800, 801, 807, 815, 819, 847,
                                     854, 858, 860, 880, 881, 883, 909, 912, 914, 919, 925, 927, 934, 950, 972, 973,
                                     997, 998]

MINI_IMAGENET_CLS_TO_IMAGENET_NAME = [
    'house_finch',
    'robin',
    'triceratops',
    'green_mamba',
    'harvestman',
    'toucan',
    'goose',
    'jellyfish',
    'nematode',
    'king_crab',
    'dugong',
    'Walker_hound',
    'Ibizan_hound',
    'Saluki',
    'golden_retriever',
    'Gordon_setter',
    'komondor',
    'boxer',
    'Tibetan_mastiff',
    'French_bulldog',
    'malamute',
    'dalmatian',
    'Newfoundland',
    'miniature_poodle',
    'white_wolf',
    'African_hunting_dog',
    'Arctic_fox',
    'lion',
    'meerkat',
    'ladybug',
    'rhinoceros_beetle',
    'ant',
    'black-footed_ferret',
    'three-toed_sloth',
    'rock_beauty',
    'aircraft_carrier',
    'ashcan',
    'barrel',
    'beer_bottle',
    'bookshop',
    'cannon',
    'carousel',
    'carton',
    'catamaran',
    'chime',
    'clog',
    'cocktail_shaker',
    'combination_lock',
    'crate',
    'cuirass',
    'dishrag',
    'dome',
    'electric_guitar',
    'file',
    'fire_screen',
    'frying_pan',
    'garbage_truck',
    'hair_slide',
    'holster',
    'horizontal_bar',
    'hourglass',
    'iPod',
    'lipstick',
    'miniskirt',
    'missile',
    'mixing_bowl',
    'oboe',
    'organ',
    'parallel_bars',
    'pencil_box',
    'photocopier',
    'poncho',
    'prayer_rug',
    'reel',
    'school_bus',
    'scoreboard',
    'slot',
    'snorkel',
    'solar_dish',
    'spider_web',
    'stage',
    'tank',
    'theater_curtain',
    'tile_roof',
    'tobacco_shop',
    'unicycle',
    'upright',
    'vase',
    'wok',
    'worm_fence',
    'yawl',
    'street_sign',
    'consomme',
    'trifle',
    'hotdog',
    'orange',
    'cliff',
    'coral_reef',
    'bolete',
    'ear'
]


SEMANTIC_SUBSETS_CLASSES = {
    "birds": [0, 1, 5, 6],
    "canidea": [20, 21, 22, 24, 26],
    "insects": [4, 29, 30, 31],
}


def get_miniimagenet_dataset(imsize=(224, 224)):

    input_transform = \
        transforms.Compose([
            transforms.Resize(imsize),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    dataset = ImageFolder(
        root=os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "miniimagenet"
        ),
        transform=input_transform,
    )
    return dataset
    return input_transform


if __name__ == "__main__":
    miniimagenet_dataset = get_miniimagenet_dataset()
    print(len(miniimagenet_dataset))
    x, y = miniimagenet_dataset[0]
    print(x.shape)
    print(y)
