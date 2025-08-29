"""Train a Siamese model using images generated on the fly."""

import math
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from src.config import load_config
from src.datasets import LabeledCombinedDataset, dataset_transform
from src.model import SharedSiamese

# * Entry Point

config = (
    load_config("config.toml")
    if len(sys.argv) < 2 or sys.argv[1] == ""
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
model = SharedSiamese(embedding_size=config["hyperparameters"]["embedding_size"]).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.TripletMarginLoss(
    margin=config["hyperparameters"]["margin"],
    p=config["hyperparameters"]["p_val"],
    swap=config["hyperparameters"]["triplet_swapping"],
)

# * Data

shoeprint_transform = dataset_transform(
    config["data"]["image_size"],
    mean=config["data"]["shoeprint_dataset_mean"],
    std=config["data"]["shoeprint_dataset_std"],
    offset=True,
    offset_translation=config["training"]["shoeprint_augmentation"]["max_translation"],
    offset_max_rotation=config["training"]["shoeprint_augmentation"]["max_rotation"],
    offset_scale_diff=config["training"]["shoeprint_augmentation"]["max_scale"],
    flip=config["training"]["shoeprint_augmentation"]["flip"],
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
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_augmented_transform,
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=config["hyperparameters"]["batch_size"],
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)

# ** Validation

val_dataset = LabeledCombinedDataset(
    config["data"]["shoeprint_data_dir"],
    config["data"]["shoemark_data_dir"],
    mode="val",
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_normal_transform,
)

# ** Testing

wvu_dataset = LabeledCombinedDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoeprints/",
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/WVU2019/Shoemarks/",
    mode="test",
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_normal_transform,
)

fid_dataset = LabeledCombinedDataset(
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoeprints/",
    "/home/struan/Vault/University/Doctorate/Data/Siamese/Testing/FID-300/Shoemarks/",
    mode="test",
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_normal_transform,
)


# * Main loop


def _write_line(line: str, pbar: tqdm, checkpoint_dir: Path):
    pbar.write(line)
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

                # Get embeddings
                shoeprint_embeddings = model(shoeprints)  # [b, d]
                shoemark_embeddings = model(shoemarks)  # [b, d]

                # Pairwise distances matrix [N, N]
                dists = torch.cdist(
                    shoeprint_embeddings, shoemark_embeddings, p=config["hyperparameters"]["p_val"]
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
                current_batch_size = shoeprint_batch.shape(0)
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

                losses += loss.item()

                if epoch % config["training"]["print_iter"] == 0 and epoch != 0:
                    line = f"Epoch {epoch} loss: {(losses / config['training']['print_iter'])}\n"
                    _write_line(line, pbar, checkpoint_dir)
                    losses = 0

                if (
                    epoch % config["training"]["val_iter"] == 0
                    or epoch == config["training"]["epochs"] - 1
                ):
                    val = evaluate(p=5, dataset=val_dataset)
                    line = f"Epoch {epoch} p5 validation: = {val}\n"
                    _write_line(line, pbar, checkpoint_dir)

                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        },
                        checkpoint_dir / f"siamese_{epoch}.tar",
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()


# * Evaluation


@torch.no_grad()
def evaluate(
    p: int = 5,
    *,
    dataset: LabeledCombinedDataset,
    checkpoint: str | Path | None = None,
):
    """Evaluate model using all shoemarks in a dataset."""
    model.eval()

    def tensor_factory():
        return torch.zeros(1)

    if checkpoint:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # pyright: ignore

    shoeprint_embeddings = []
    shoemark_embeddings = defaultdict(tensor_factory)

    for i, (shoeprint, shoemarks) in enumerate(dataset):
        shoeprint_embedding = model(shoeprint.unsqueeze(0).to(device)).cpu()
        shoeprint_embeddings.append(shoeprint_embedding.squeeze())

        # Not as fast as batching all shoemarks but works for very large numbers of shoemarks
        if len(shoemarks) > 0:
            shoemark_embeddings[i] = torch.cat(
                [model(shoemark.unsqueeze(0).to(device)).cpu() for shoemark in shoemarks]
            )

    shoeprint_embeddings = torch.stack(shoeprint_embeddings)

    model.train()

    ranks = []
    for shoe_id, class_shoemark_embeddings in shoemark_embeddings.items():
        # Compare distance between shoemark embedding and _all_ shoeprint embeddings

        dists = torch.cdist(
            class_shoemark_embeddings, shoeprint_embeddings, p=config["hyperparameters"]["p_val"]
        )

        # Get indices of distances sorted small->large
        sorted_dists = torch.argsort(dists)

        shoemark_ranks = (sorted_dists == shoe_id).nonzero()

        ranks += [t.item() for t in shoemark_ranks[:, 1]]

    ranks = np.array(ranks)

    k = math.ceil(max(1, len(shoeprint_embeddings) * p / 100))
    return np.mean(ranks <= k)


# * Entry Point

if __name__ == "__main__":
    training_loop()

# Local Variables:
# jinx-local-words: "noqa"
# End:
