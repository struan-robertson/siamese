"""Train a Siamese model using images generated on the fly."""

import math
import random
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from src.config import load_config
from src.datasets import (
    LabeledCombinedDataset,
    LabeledIndividualDataset,
    dataset_transform,
)
from src.model import BottleneckClassification, ShorterClassification
from tqdm import tqdm

# * Config


config = (
    load_config("config.toml")
    if len(sys.argv) < 2 or sys.argv[1] == "" or sys.argv[1] == "-i"
    else load_config(sys.argv[1])
)


# * Seeding


def seed_worker(worker_id):
    """Seed DataLoader workers with random seed."""
    worker_seed = (
        config["training"]["seed"] + worker_id
    ) % 2**32  # Ensure we don't overflow 32 bit
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)

    # Passed to dataloaders
    dataloader_g = torch.Generator()
    dataloader_g.manual_seed(config["training"]["seed"])


torch.manual_seed(config["training"]["seed"])
np.random.default_rng(config["training"]["seed"])
random.seed(config["training"]["seed"])


# * PyTorch


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# model = ShorterClassification().to(device)
model = BottleneckClassification().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)

criterion = torch.nn.BCEWithLogitsLoss()


# * Data

shoeprint_augmented_transform = dataset_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoeprint_dataset_mean"],
    std=config["data"]["shoeprint_dataset_std"],
    offset=True,
    offset_translation=config["training"]["shoeprint_augmentation"]["max_translation"],
    offset_max_rotation=config["training"]["shoeprint_augmentation"]["max_rotation"],
    offset_scale_diff=config["training"]["shoeprint_augmentation"]["max_scale"],
    flip=config["training"]["shoeprint_augmentation"]["flip"],
)

shoeprint_normal_transform = dataset_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoeprint_dataset_mean"],
    std=config["data"]["shoeprint_dataset_std"],
    offset=False,
    flip=False,
)

shoemark_augmented_transform = dataset_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoemark_dataset_mean"],
    std=config["data"]["shoemark_dataset_std"],
    offset=True,
    offset_translation=config["training"]["shoemark_augmentation"]["max_translation"],
    offset_max_rotation=config["training"]["shoemark_augmentation"]["max_rotation"],
    offset_scale_diff=config["training"]["shoemark_augmentation"]["max_scale"],
    flip=config["training"]["shoemark_augmentation"]["flip"],
)

shoemark_normal_transform = dataset_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoemark_dataset_mean"],
    std=config["data"]["shoemark_dataset_std"],
    offset=False,
    flip=False,
)


# ** Training

dataset = LabeledCombinedDataset(
    config["data"]["shoeprint_data_dir"],
    config["data"]["shoemark_data_dir"],
    mode="train",
    shoeprint_transform=shoeprint_augmented_transform,
    shoemark_transform=shoemark_augmented_transform,
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["hyperparameters"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=True,
    drop_last=False,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)

# ** Validation

shoeprint_val_dataset = LabeledIndividualDataset(
    config["data"]["shoeprint_data_dir"],
    mode="val",
    transform=shoeprint_normal_transform,
)

shoemark_val_dataset = LabeledIndividualDataset(
    config["data"]["shoemark_data_dir"],
    mode="val",
    transform=shoemark_normal_transform,
)

# ** Testing

wvu_shoeprint_dataset = LabeledIndividualDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoeprints/",
    mode="test",
    transform=shoeprint_normal_transform,
)

wvu_shoemark_dataset = LabeledIndividualDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoemarks/",
    mode="test",
    transform=shoemark_normal_transform,
)

fid_shoeprint_dataset = LabeledIndividualDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoeprints/",
    mode="test",
    transform=shoeprint_normal_transform,
)

fid_shoemark_dataset = LabeledIndividualDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoemarks/",
    mode="test",
    transform=shoemark_normal_transform,
)


# * Main loop


def _write_line(line: str, pbar: tqdm, checkpoint_dir: Path):
    pbar.write(line, end="")
    with (checkpoint_dir / "siamese.log").open("a") as f:
        f.write(line)


# Find negatives closer to the anchor than positives
# Violating d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
def training_loop():
    """Run training loop for siamese model."""
    checkpoint_dir = Path("checkpoints") / config["training"]["name"]
    checkpoint_dir.mkdir()

    with tqdm(total=config["training"]["epochs"], dynamic_ncols=True) as pbar:
        for epoch in range(config["training"]["epochs"]):
            pbar.set_description(f"Epoch: {epoch}")
            losses = 0

            for shoeprint_batch, shoemark_batch, label_batch in loader:
                shoeprints = shoeprint_batch.to(device)
                shoemarks = shoemark_batch.to(device)
                labels = label_batch.to(device)

                output = model(shoeprints, shoemarks).squeeze()

                # Calculate BCE loss
                loss = criterion(output, labels)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                losses += loss.item()

            if epoch % config["training"]["print_iter"] == 0 and epoch != 0:
                line = f"Epoch {epoch} loss: {(losses / config['training']['print_iter'])}\n"
                _write_line(line, pbar, checkpoint_dir)
                losses = 0

            if (
                epoch % config["training"]["val_iter"] == 0
                or epoch == config["training"]["epochs"] - 1
            ) and epoch != 0:
                val = evaluate(
                    p=5,
                    shoeprint_dataset=shoeprint_val_dataset,
                    shoemark_dataset=shoemark_val_dataset,
                )
                line = f"Epoch {epoch} p5 validation: = {val}\n"
                _write_line(line, pbar, checkpoint_dir)

                torch.save(
                    {
                        "model_state_dict": model.state_dict(),
                        "optim_state_dict": optimizer.state_dict(),
                    },
                    checkpoint_dir / f"siamese_{epoch}.tar",
                )

            pbar.update()


# * Evaluation


# TODO refactor for quicker evaluation
@torch.no_grad()
def evaluate(
    p: int = 5,
    *,
    shoeprint_dataset: LabeledIndividualDataset,
    shoemark_dataset: LabeledIndividualDataset,
    checkpoint: str | Path | None = None,
    move_failures: bool = False,
):
    """Evaluate model using all shoemarks in a dataset."""
    model.eval()

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # pyright: ignore

    shoeprint_loader = torch.utils.data.DataLoader(
        shoeprint_dataset,
        batch_size=config["inference"]["batch_size"],
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # Load all shoeprints with their classes
    shoeprint_tensors = []
    shoeprint_classes = []

    for batch in shoeprint_loader:
        (classes, _), images = batch
        shoeprint_tensors.append(images)
        shoeprint_classes.extend(classes)

    k = math.ceil(max(1, len(shoeprint_classes) * p / 100))
    ranks = []

    # Compare every shoemark against every shoeprint and see where it ranks
    for (shoemark_class, shoemark_id), shoemark in tqdm(
        shoemark_dataset, desc="Evaluating: "
    ):
        probabilities = torch.stack(
            [
                model(
                    shoeprints.to(device),
                    shoemark.to(device).expand(len(shoeprints), *shoemark.shape),
                )
                .cpu()
                .squeeze()
                for shoeprints in shoeprint_tensors
            ]
        ).flatten()

        sorted_probabilities = torch.argsort(probabilities)
        correct_idx = shoeprint_classes.index(shoemark_class)

        shoemark_rank = (sorted_probabilities == correct_idx).nonzero().item()

        ranks.append(shoemark_rank)

        if move_failures and shoemark_rank > k:
            shutil.copy(
                config["data"]["shoemark_data_dir"]
                / "val"
                / f"{shoemark_class}_{shoemark_id}.png",
                "failed_val/",
            )

    model.train()

    ranks = np.array(ranks)

    return np.mean(ranks <= k)


# * Entry Point

if __name__ == "__main__":
    training_loop()

# Local Variables:
# jinx-local-words: "noqa"
# End:
