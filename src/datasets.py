"""Load datasets using torch.utils.data.Dataset."""

import math
import random
import re
from pathlib import Path
from typing import Literal

import torch
import torchvision.transforms.functional as F
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from tqdm import tqdm

_dataset_mode = Literal["train", "test", "val"]


def calculate_stats(loader: torch.utils.data.DataLoader):
    """Calculate per-channel mean and std using explicit sum of squares."""
    num_channels = loader.dataset[0][0].shape[0]
    sum_pixels = torch.zeros(num_channels)
    sum_squares = torch.zeros(num_channels)
    total_pixels = 0

    for shoeprints, shoemarks in tqdm(loader):
        # Mean over batch, height and width, but not over channels

        if len(shoemarks.shape) == 5:
            reshaped_shoemarks = shoemarks.reshape(-1, *shoemarks.shape[-3:])
        else:
            # Don't overwrite loop variable
            reshaped_shoemarks = shoemarks

        batched_tensors = torch.cat([shoeprints, reshaped_shoemarks], dim=0)
        batched_tensors = batched_tensors.flatten(start_dim=2)  # [B, C, H*W]

        # Accumulate statistics
        sum_pixels += batched_tensors.sum(dim=(0, 2))
        sum_squares += (batched_tensors**2).sum(dim=(0, 2))
        total_pixels += batched_tensors.shape[0] * batched_tensors.shape[2]

    # Final calculations
    mean = sum_pixels / total_pixels
    std = torch.sqrt((sum_squares / total_pixels) - (mean**2))

    return mean, std


def no_norm_transform(
    image_size: tuple[int, int],
):
    """Initialise transforms with no normalisations, used for calculating dataset stats."""
    return transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
        ]
    )


def dataset_transform(
    image_size: tuple[int, int],
    *,
    mean: float | tuple[float, float, float],
    std: float | tuple[float, float, float],
    offset: bool = False,
):
    """Initialise transforms for a dataset."""
    transform_list = [
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]

    if offset:
        transform_list.append(RandomOffsetTransormation())

    return transforms.Compose(transform_list)


class RandomOffsetTransormation:
    """Randomly shift the image in any direction and rotate."""

    def __init__(self, offset: int = 128, max_rotation: int = 10, scale_diff: float = 0.25):
        self.offset = offset
        self.max_rotation = max_rotation
        self.scale_diff = scale_diff

    def __call__(self, img):
        angle_rad = torch.rand(1).item() * 2 * math.pi
        dx = int(self.offset * math.cos(angle_rad))
        dy = int(self.offset * math.sin(angle_rad))
        rotation = torch.empty(1).uniform_(-self.max_rotation, self.max_rotation).item()
        scale = torch.empty(1).uniform_(1 - self.scale_diff, 1 + self.scale_diff).item()

        return F.affine(
            img,
            angle=rotation,
            translate=[dx, dy],
            scale=scale,
            shear=0.0,  # pyright: ignore [reportArgumentType]
            fill=1.0,  # pyright: ignore [reportArgumentType]
        )


class LabeledCombinedDataset(Dataset):
    """Load shoeprint and shoemark images. Returns (shoeprint, (shoemarks)) tuples."""

    def __init__(
        self,
        shoeprint_path: Path | str,
        shoemark_path: Path | str,
        *,
        mode: _dataset_mode,
        shoeprint_transform,
        shoemark_transform,
        all_shoemarks: bool = False,
    ):
        shoeprint_path = Path(shoeprint_path).expanduser() / mode
        shoemark_path = Path(shoemark_path).expanduser() / mode

        shoeprint_jpg_files = list(shoeprint_path.rglob("*.jpg"))
        shoeprint_png_files = list(shoeprint_path.rglob("*.png"))
        self.shoeprint_files = shoeprint_jpg_files + shoeprint_png_files

        shoemark_jpg_files = list(shoemark_path.rglob("*.jpg"))
        shoemark_png_files = list(shoemark_path.rglob("*.png"))
        self.shoemark_files = {f.stem: f for f in shoemark_jpg_files + shoemark_png_files}

        self.shoeprint_transform = shoeprint_transform
        self.shoemark_transform = shoemark_transform
        self.mode = mode
        self.all_shoemarks = all_shoemarks

    def __len__(self):
        return len(self.shoeprint_files)

    def __getitem__(self, idx: int):
        shoeprint = self.shoeprint_files[idx]
        shoeprint_name = shoeprint.stem
        shoeprint_image = Image.open(shoeprint).convert("RGB")

        # Used for matching shoemark filenames to shoeprints
        pattern = re.escape(shoeprint_name) + r"_\d+$"

        if self.mode == "train" and not self.all_shoemarks:
            shoemark_file = random.choice(
                [file for key, file in self.shoemark_files.items() if re.fullmatch(pattern, key)]
            )
            shoemark = self.shoemark_transform(Image.open(shoemark_file).convert("RGB"))
        # Used for calculating dataset statistics
        elif self.mode == "train" and self.all_shoemarks:
            shoemark_files = [
                self.shoemark_transform(Image.open(file).convert("RGB"))
                for key, file in self.shoemark_files.items()
                if re.fullmatch(pattern, key)
            ]
            shoemark = torch.stack(shoemark_files)
        else:
            shoemark_files = [
                file for key, file in self.shoemark_files.items() if re.fullmatch(pattern, key)
            ]
            shoemark = self.shoemark_transform(Image.open(shoemark_files[0]).convert("RGB"))

        shoeprint = self.shoeprint_transform(shoeprint_image)

        return shoeprint, shoemark
