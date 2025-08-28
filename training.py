"""Train a Siamese model using images generated on the fly."""

import math
import random
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.datasets import LabeledCombinedDataset, dataset_transform
from src.model import SharedSiamese

# * Config

seed = 4242
pre_trained = False

p_val = 2
margin = 0.5
batch_size = 32
image_size = (512, 256)

shoeprint_data_dir = "/home/struan/Vault/University/Doctorate/Data/Siamese/Shoeprints"
shoeprint_dataset_mean = (0.8864, 0.8864, 0.8864)
shoeprint_dataset_std = (0.2424, 0.2424, 0.2424)

shoemark_data_dir = "/home/struan/Vault/University/Doctorate/Data/Siamese/Shoemarks"
shoemark_dataset_mean = (0.6739, 0.6194, 0.5622)
shoemark_dataset_std = (0.2489, 0.2591, 0.2871)

test_shoeprint_data_dir = "/home/struan/Datasets/WVU2019 Cropped/Gallery"
test_shoemark_data_dir = "/home/struan/Datasets/WVU2019 Cropped/Query"

# * Seeding


def seed_worker(worker_id):
    """Seed DataLoader workers with random seed."""
    worker_seed = (seed + worker_id) % 2**32  # Ensure we don't overflow 32 bit
    np.random.default_rng(worker_seed)
    random.seed(worker_seed)

    # Passed to dataloaders
    dataloader_g = torch.Generator()
    dataloader_g.manual_seed(seed)


torch.manual_seed(seed)
np.random.default_rng(seed)
random.seed(seed)

# * PyTorch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = SharedSiamese().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
criterion = torch.nn.TripletMarginLoss(margin=margin, p=p_val, swap=True)

# * Data

shoeprint_transform = dataset_transform(
    image_size, mean=shoeprint_dataset_mean, std=shoeprint_dataset_std, flip=False
)

shoemark_transform = dataset_transform(
    image_size, mean=shoemark_dataset_mean, std=shoemark_dataset_std, flip=True
)


# ** Training

dataset = LabeledCombinedDataset(
    shoeprint_data_dir,
    shoemark_data_dir,
    mode="train",
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_transform,
)

loader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
    worker_init_fn=seed_worker,
    persistent_workers=True,
)

# ** Validation

val_dataset = LabeledCombinedDataset(
    shoeprint_data_dir,
    shoemark_data_dir,
    mode="val",
    shoeprint_transform=shoeprint_transform,
    shoemark_transform=shoemark_transform,
)

val_dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=False,
    drop_last=False,
)

# * Main loop


def _write_line(line: str, pbar: tqdm):
    pbar.write(line)
    with Path("checkpoints/siamese.log").open("a") as f:
        f.write(line)


# Find negatives closer to the anchor than positives
# Violating d(anchor, positive) < d(anchor, negative) < d(anchor, positive) + margin
def training_loop(steps: int, print_iter: int, val_iter: int, save_iter: int):
    """Run training loop for siamese model."""
    epochs = math.ceil(steps / len(dataset))

    step = 0

    with tqdm(total=(epochs * len(dataset)) // batch_size, dynamic_ncols=True) as pbar:
        for epoch in range(epochs):
            pbar.set_description(f"Epoch: {epoch}")
            losses = 0

            for shoeprint_batch, shoemark_batch in loader:
                shoeprints = shoeprint_batch.to(device)
                shoemarks = shoemark_batch.to(device)

                # Get embeddings
                shoeprint_embeddings = model(shoeprints)  # [b, d]
                shoemark_embeddings = model(shoemarks)  # [b, d]

                # Pairwise distances matrix [N, N]
                dists = torch.cdist(shoeprint_embeddings, shoemark_embeddings, p=p_val)

                # Positive distances
                pos_dists = dists.diag().view(-1, 1)

                # Mask to exclude the positive pairs (0s everywhere apart from the diagonal)
                idt_mask = torch.eye(pos_dists.size(0), dtype=torch.bool, device=device)

                # Identify semi-hard violations
                semi_hard_mask = (dists > pos_dists) & (dists < pos_dists + margin)
                semi_hard_mask[idt_mask] = False

                # Store indices of selected negatives
                neg_idxs = []
                for i in range(batch_size):
                    violation_inds = torch.where(semi_hard_mask[i])[0]

                    if len(violation_inds) > 0:
                        # Get hardest violation
                        hardest_violation_idx = violation_inds[
                            torch.argmin(dists[i, violation_inds])
                        ]
                        neg_idxs.append(hardest_violation_idx.item())
                    else:
                        # Ensure not to select the positive
                        candidates = [j for j in range(batch_size) if j != i]
                        neg_idxs.append(random.choice(candidates))

                # Convert to tensor indices
                neg_idxs = torch.tensor(neg_idxs, device=device)

                # Extract negative embeddings
                negatives = shoemark_embeddings[neg_idxs]

                # Calculate triplet loss
                loss = criterion(shoeprint_embeddings, shoemark_embeddings, negatives)

                losses += loss.item()

                if step % print_iter == 0 and step != 0:
                    line = f"Step {step} loss: {(losses / print_iter)}\n"
                    _write_line(line, pbar)
                    losses = 0

                if step % val_iter == 0:
                    val = validate()
                    line = f"Step {step} p5 validation: = {val}\n"
                    _write_line(line, pbar)

                if step % save_iter == 0 and step != 0:
                    torch.save(
                        {
                            "model_state_dict": model.state_dict(),
                            "optim_state_dict": optimizer.state_dict(),
                        },
                        f"checkpoints/siamese_{step}.tar",
                    )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.update()
                step += 1

    val = validate()
    line = f"Step {step} p5 validation: = {val}\n"
    _write_line(line, pbar)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optim_state_dict": optimizer.state_dict(),
        },
        f"checkpoints/siamese_{step}.tar",
    )


# * Validation


@torch.no_grad()
def validate(p: int = 5):
    """Evaluate model on val set."""
    model.eval()

    # This function is much simpler than test() as it only accounts for one shoemark per shoeprint
    # TODO use all shoemarks per shoeprint
    shoeprint_embeddings, shoemark_embeddings = [], []
    for shoeprint, shoemark in val_dataloader:
        shoeprint_embeddings.append(model(shoeprint.to(device)).cpu())
        shoemark_embeddings.append(model(shoemark.to(device)).cpu())

    model.train()

    shoeprint_embeddings = torch.cat(shoeprint_embeddings)  # [N, d]
    shoemark_embeddings = torch.cat(shoemark_embeddings)  # [N, d]

    # Pairwise distances matrix [N, N]
    dists = torch.cdist(shoeprint_embeddings, shoemark_embeddings, p=p_val)

    # Get ranks (position of correct match in sorted distances)
    # The diagonal here gives the distance between corresponding shoeprints and shoemarks
    ranks = (dists.diag().unsqueeze(1) >= dists).sum(dim=1).float()

    # Calculate top-p% accuracy
    k = max(1, int(len(dataset) * p / 100))

    return (ranks <= k).float().mean().item()


# * Testing


@torch.no_grad()
def test(p: int = 5, checkpoint: str | None = None):
    """Evaluate model on test set, optionally loading from a checkpoint."""
    model.eval()

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        model.load_state_dict(checkpoint["model_state_dict"])  # pyright: ignore

    shoeprint_path = Path(test_shoeprint_data_dir)
    shoemark_path = Path(test_shoemark_data_dir)

    shoeprint_files = list(shoeprint_path.rglob("*.png"))
    shoemark_files = list(shoemark_path.rglob("*.png"))

    def calc_embedding(f: Path, print_type: Literal["shoeprint", "shoemark"]):
        i = Image.open(f).convert("RGB")  # Not all test sets are RGB

        t = (
            shoeprint_transform(i).to(device)  # pyright: ignore [reportAttributeAccessIssue]
            if print_type == "shoeprint"
            else shoemark_transform(i).to(device)  # pyright: ignore [reportAttributeAccessIssue]
        )

        return model(t.unsqueeze(0)).cpu()

    # There are many potential shoemarks for each shoeprint
    shoeprint_embeddings = {
        f.stem[:3]: calc_embedding(f, "shoeprint").squeeze() for f in shoeprint_files
    }
    shoeprint_ids = list(shoeprint_embeddings.keys())
    shoeprint_embeddings = torch.stack(list(shoeprint_embeddings.values()))  # Create tensor

    shoemark_embeddings = {f.stem: calc_embedding(f, "shoemark").squeeze() for f in shoemark_files}

    model.train()

    ranks = []
    for shoe_id, shoemark_embedding in shoemark_embeddings.items():
        # Compare distance between shoemark embedding and _all_ shoeprint embeddings
        dists = torch.cdist(
            shoemark_embedding.unsqueeze(0), shoeprint_embeddings, p=p_val
        ).squeeze()

        # Sort indices of distances sorted small->large
        sorted_dists = torch.argsort(dists)

        # Get the index of the correct shoeprint
        correct_idx = shoeprint_ids.index(shoe_id[:3])

        # Get the position of the correct index
        rank = (sorted_dists == int(correct_idx)).nonzero().item()

        ranks.append(rank)

    ranks = np.array(ranks)

    k = max(1, int(len(shoeprint_files) * p / 100))
    return np.mean(ranks <= k)


if __name__ == "__main__":
    # checkpoint = torch.load("checkpoints/siamese_225.tar")

    # model.load_state_dict(checkpoint["model_state_dict"])
    # optimizer.load_state_dict(checkpoint["optim_state_dict"])

    training_loop(steps=100_000, print_iter=50, val_iter=500, save_iter=500)
