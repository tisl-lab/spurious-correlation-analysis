from datasets import mini_imagenet

from .colored_mnist import ColoredMNIST
from .colored_cifar10 import ColoredCIFAR10
from .spawrious224 import Spawrious224
from .celeba import CelebA
# from .mini_imagenet import mini_imagenet

__all__ = ["ColoredMNIST", "ColoredCIFAR10", "Spawrious224", "CelebA", "get_miniimagenet_dataset"]
