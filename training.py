"""Train a Siamese model using images generated on the fly."""

import math
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from src.config import load_config
from src.datasets import LabeledCombinedDataset, gpu_transform
from src.model import SharedSiamese
from torchvision import transforms
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

if torch.cuda.is_available():
    torch.cuda.manual_seed(config["training"]["seed"])
    torch.cuda.manual_seed_all(config["training"]["seed"])

# * PyTorch


device = torch.device(
    f"cuda:{config['training']['gpu_number']}" if torch.cuda.is_available() else "cpu"
)

shoeprint_model = SharedSiamese(
    embedding_size=config["hyperparameters"]["embedding_size"],
    pre_trained=config["training"]["pre_training"]["pre_trained"],
    refreeze=config["training"]["pre_training"]["refreeze"],
    permafrost=config["training"]["pre_training"]["permafrost"],
).to(device)

shoemark_model = SharedSiamese(
    embedding_size=config["hyperparameters"]["embedding_size"],
    pre_trained=config["training"]["pre_training"]["pre_trained"],
    refreeze=config["training"]["pre_training"]["refreeze"],
    permafrost=config["training"]["pre_training"]["permafrost"],
).to(device)

shoeprint_optimizer = torch.optim.AdamW(
    shoeprint_model.parameters(), lr=0.001, weight_decay=1e-4
)
shoemark_optimizer = torch.optim.AdamW(
    shoemark_model.parameters(), lr=0.001, weight_decay=1e-4
)

criterion = torch.nn.TripletMarginLoss(
    margin=config["hyperparameters"]["margin"],
    p=config["hyperparameters"]["p_val"],
    swap=config["hyperparameters"]["triplet_swapping"],
)

# * Data

shoeprint_augmented_transform = gpu_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoeprint_dataset_mean"],
    std=config["data"]["shoeprint_dataset_std"],
    offset=True,
    offset_translation=config["training"]["shoeprint_augmentation"]["max_translation"],
    offset_max_rotation=config["training"]["shoeprint_augmentation"]["max_rotation"],
    offset_scale_diff=config["training"]["shoeprint_augmentation"]["max_scale"],
    flip=config["training"]["shoeprint_augmentation"]["flip"],
)

shoeprint_normal_transform = gpu_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoeprint_dataset_mean"],
    std=config["data"]["shoeprint_dataset_std"],
    offset=False,
    flip=False,
)

shoemark_augmented_transform = gpu_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoemark_dataset_mean"],
    std=config["data"]["shoemark_dataset_std"],
    offset=True,
    offset_translation=config["training"]["shoemark_augmentation"]["max_translation"],
    offset_max_rotation=config["training"]["shoemark_augmentation"]["max_rotation"],
    offset_scale_diff=config["training"]["shoemark_augmentation"]["max_scale"],
    flip=config["training"]["shoemark_augmentation"]["flip"],
)

shoemark_normal_transform = gpu_transform(
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
    shoeprint_transform=transforms.ToTensor(),
    shoemark_transform=transforms.ToTensor(),
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

val_dataset = LabeledCombinedDataset(
    config["data"]["shoeprint_data_dir"],
    config["data"]["shoemark_data_dir"],
    mode="val",
    shoeprint_transform=transforms.ToTensor(),
    shoemark_transform=transforms.ToTensor(),
)

# ** Testing

# TODO if this works then clean up adding transforms here
wvu_dataset = LabeledCombinedDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoeprints/",
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoemarks/",
    mode="test",
    shoeprint_transform=transforms.ToTensor(),
    shoemark_transform=transforms.ToTensor(),
)

fid_dataset = LabeledCombinedDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoeprints/",
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoemarks/",
    mode="test",
    shoeprint_transform=transforms.ToTensor(),
    shoemark_transform=transforms.ToTensor(),
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

            for shoeprint_batch, shoemark_batch in loader:
                shoeprints = shoeprint_batch.to(device)
                shoemarks = shoemark_batch.to(device)

                # Transform on GPU
                shoeprints = shoeprint_augmented_transform(shoeprints)
                shoemarks = shoemark_augmented_transform(shoemarks)

                # Get embeddings
                shoeprint_embeddings = shoeprint_model(shoeprints)  # [b, d]
                shoemark_embeddings = shoemark_model(shoemarks)  # [b, d]

                # Pairwise distances matrix [N, N]
                dists = torch.cdist(
                    shoeprint_embeddings,
                    shoemark_embeddings,
                    p=config["hyperparameters"]["p_val"],
                )

                # Positive distances
                pos_dists = dists.diag().view(-1, 1)

                # Mask to exclude the positive pairs (0s everywhere apart from the diagonal)
                idt_mask = torch.eye(pos_dists.size(0), dtype=torch.bool, device=device)

                # Identify semi-hard violations
                semi_hard_mask = (dists > pos_dists) & (
                    dists < pos_dists + config["hyperparameters"]["margin"]
                )
                semi_hard_mask[idt_mask] = False

                # Store indices of selected negatives
                neg_idxs = []
                # As we don't drop the last batch, this may be less than overall batch size
                current_batch_size = shoeprint_batch.shape[0]
                for i in range(current_batch_size):
                    violation_inds = torch.where(semi_hard_mask[i])[0]

                    if len(violation_inds) > 0:
                        # Get hardest violation
                        hardest_violation_idx = violation_inds[
                            torch.argmin(dists[i, violation_inds])
                        ]
                        neg_idxs.append(hardest_violation_idx.item())
                    else:
                        # Ensure not to select the positive
                        candidates = [j for j in range(current_batch_size) if j != i]
                        neg_idxs.append(random.choice(candidates))

                # Convert to tensor indices
                neg_idxs = torch.tensor(neg_idxs, device=device)

                # Extract negative embeddings
                negatives = shoemark_embeddings[neg_idxs]

                # Calculate triplet loss
                loss = criterion(shoeprint_embeddings, shoemark_embeddings, negatives)

                shoeprint_optimizer.zero_grad()
                shoemark_optimizer.zero_grad()
                loss.backward()
                shoeprint_optimizer.step()
                shoemark_optimizer.step()

                losses += loss.item()

            if epoch % config["training"]["print_iter"] == 0 and epoch != 0:
                line = f"Epoch {epoch} loss: {(losses / config['training']['print_iter'])}\n"
                _write_line(line, pbar, checkpoint_dir)
                losses = 0

            if (
                epoch % config["training"]["val_iter"] == 0
                or epoch == config["training"]["epochs"] - 1
            ) and epoch != 0:
                val = evaluate(p=5, dataset=val_dataset)
                line = f"Epoch {epoch} p5 validation: = {val}\n"
                _write_line(line, pbar, checkpoint_dir)

                torch.save(
                    {
                        "shoeprint_model_state_dict": shoeprint_model.state_dict(),
                        "shoemark_model_state_dict": shoemark_model.state_dict(),
                        "shoeprint_optim_state_dict": shoeprint_optimizer.state_dict(),
                        "shoemark_optim_state_dict": shoemark_optimizer.state_dict(),
                    },
                    checkpoint_dir / f"siamese_{epoch}.tar",
                )

            # TODO tidy this up a bit
            if (
                config["training"]["pre_training"]["pre_trained"]
                and config["training"]["pre_training"]["frozen"]
                and epoch != 0
                and (epoch - config["training"]["pre_training"]["defrost"])
                % config["training"]["pre_training"]["epoch_unfreeze"]
                == 0
            ):
                shoeprint_model.unfreeze_next()
                shoemark_model.unfreeze_next()

            pbar.update()


# * Evaluation


@torch.no_grad()
def evaluate(
    p: int = 5,
    *,
    dataset: LabeledCombinedDataset,
    checkpoint: str | Path | None = None,
    move_failures: bool = False,
):
    """Evaluate model using all shoemarks in a dataset."""
    shoeprint_model.eval()
    shoemark_model.eval()

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # pyright: ignore

    shoeprint_embeddings = defaultdict(lambda: torch.zeros(1))
    shoemark_embeddings = defaultdict(lambda: torch.zeros(1))

    for shoeprint_class, (shoeprint, shoemarks) in dataset:
        normalised_shoeprint = shoeprint_normal_transform(shoeprint.to(device))
        shoeprint_embedding = shoeprint_model(normalised_shoeprint.unsqueeze(0)).cpu()
        shoeprint_embeddings[shoeprint_class] = shoeprint_embedding.squeeze()

        # Not as fast as batching all shoemarks but works for very large numbers of shoemarks
        if len(shoemarks) > 0:
            shoemark_embeddings[shoeprint_class] = torch.cat(
                [
                    shoemark_model(
                        shoemark_normal_transform(shoemark.to(device)).unsqueeze(0)
                    ).cpu()
                    for shoemark in shoemarks
                ]
            )

    shoeprint_class_idxs = list(shoeprint_embeddings.keys())
    shoeprint_embeddings = torch.stack(list(shoeprint_embeddings.values()))

    shoeprint_model.train()
    shoemark_model.train()

    k = math.ceil(max(1, len(shoeprint_embeddings) * p / 100))

    ranks = []
    for shoe_id, class_shoemark_embeddings in shoemark_embeddings.items():
        # Compare distance between shoemark embedding and _all_ shoeprint embeddings

        dists = torch.cdist(
            class_shoemark_embeddings,
            shoeprint_embeddings,
            p=config["hyperparameters"]["p_val"],
        )

        # Get indices of distances sorted small->large
        sorted_dists = torch.argsort(dists)

        correct_idx = shoeprint_class_idxs.index(shoe_id)
        shoemark_ranks = (sorted_dists == correct_idx).nonzero()

        for shoemark_id, rank in enumerate([t.item() for t in shoemark_ranks[:, 1]]):
            ranks.append(rank)
            if move_failures and rank > k:
                shutil.copy(
                    config["data"]["shoemark_data_dir"]
                    / "val"
                    / f"{shoe_id}_{shoemark_id}.png",
                    "failed_val/",
                )

    ranks = np.array(ranks)

    return np.mean(ranks <= k)


# * Entry Point

if __name__ == "__main__":
    training_loop()

# Local Variables:
# jinx-local-words: "noqa"
# End:
