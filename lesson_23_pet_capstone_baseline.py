import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    mobilenet_v3_small,
)

SEED = 42
BATCH_SIZE = 16
EPOCHS = 3
LEARNING_RATE = 1e-3
VAL_FRACTION = 0.20

DATA_DIR = Path("data")
RUN_DIR = Path("runs/pet_breeds_baseline")
CHECKPOINT_PATH = RUN_DIR / "best-model.pt"
METRICS_PATH = RUN_DIR / "metrics.csv"


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_loaders():
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()

    dataset = OxfordIIITPet(
        root=DATA_DIR, split="trainval", target_types="category", download=True, transform=transform
    )

    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    num_val = round(len(dataset) * VAL_FRACTION)
    train_indices = indices[:-num_val]
    val_indices = indices[-num_val:]

    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)

    train_loader = DataLoader(
        train_set,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        generator=torch.Generator().manual_seed(SEED + 2),
    )

    return train_loader, val_loader, list(dataset.classes), train_indices, val_indices


def build_model(num_classes: int) -> nn.Module:
    model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, num_classes)
    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.eval()
    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad()
        logits = model(images)
        loss = loss_function(logits, labels)
        loss.backward()
        optimizer.step()
        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size
    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    with torch.inference_mode():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = loss_function(logits, labels)
            predictions = logits.argmax(dim=1)
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (predictions == labels).sum().item()
            total_samples += batch_size

        return total_loss / total_samples, total_correct / total_samples


def save_checkpoint(
    model: nn.Module,
    class_names: list[str],
    train_indices: list[int],
    val_indices: list[int],
    epoch: int,
    val_loss: float,
    val_accuracy: float,
):
    cpu_state = {
        key: value.detach().cpu().clone()
        for key, value in model.state_dict().items()
    }

    torch.save(
        {
            "checkpoint_version": 1,
            "architecture": "mobilenet_v3_small",
            "num_classes": len(class_names),
            "class_names": class_names,
            "preprocessing": "MobileNet_V3_Small_Weights.DEFAULT.transforms()",
            "split_seed": SEED,
            "train_indices": train_indices,
            "val_indices": val_indices,
            "epoch": epoch,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "model_state_dict": cpu_state,
        },
        CHECKPOINT_PATH,
    )


def main() -> None:
    torch.manual_seed(SEED)
    device = get_device()
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    print("Device:", device)

    train_loader, val_loader, classes, train_idx, val_idx = make_loaders()
    print("Train images:", len(train_idx))
    print("Validation images:", len(val_idx))
    print("Pet breeds:", len(classes))
    print("Official test split: untouched")

    model = build_model(len(classes)).to(device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Trainable parameters:", trainable)

    # Inspect a deterministic validation batch without advancing training shuffle.
    images, labels = next(iter(val_loader))
    # inference_mode disables autograd, but only eval freezes BatchNorm/dropout.
    model.eval()
    with torch.inference_mode():
        shape = tuple(model(images.to(device)).shape)
    print("Image batch:", tuple(images.shape))
    print("Labels:", tuple(labels.shape))
    print("Logits:", shape)
    assert shape == (len(labels), len(classes))

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.classifier[-1].parameters(), lr=LEARNING_RATE)
    best_loss = float("inf")
    history: list[dict[str, float | int]] = []

    for epoch in range(1, EPOCHS + 1):
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device)
        val_loss, val_accuracy = evaluate(
            model, val_loader, loss_function, device)
        history.append(
            {"epoch": epoch,
             "train_loss": train_loss,
             "val_loss": val_loss,
             "val_accuracy": val_accuracy, }
        )

        if val_loss < best_loss:
            best_loss = val_loss
            save_checkpoint(model, classes, train_idx, val_idx, epoch, val_loss, val_accuracy)

        print(
            f"Epoch {epoch:02d}/{EPOCHS}: "
            f"train_loss={train_loss:.4f} val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    with METRICS_PATH.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file, fieldnames=["epoch", "train_loss", "val_loss", "val_accuracy"]
        )
        writer.writeheader()
        writer.writerows(history)

    # Verify the persisted artifact can be read with restricted loading.
    saved = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    assert len(saved["class_names"]) == saved["num_classes"]
    assert set(saved["train_indices"]).isdisjoint(saved["val_indices"])
    restored = mobilenet_v3_small(weights=None)
    restored.classifier[-1] = nn.Linear(
        restored.classifier[-1].in_features, saved["num_classes"]
    )
    restored.load_state_dict(saved["model_state_dict"])
    restored = restored.to(device).eval()
    with torch.inference_mode():
        restored_logits = restored(images.to(device))
    assert restored_logits.shape == (len(labels), saved["num_classes"])

    print("\nBest epoch:", saved["epoch"])
    print("Best validation loss:", round(saved["validation_loss"], 4))
    print("Best validation accuracy:", round(saved["validation_accuracy"], 4))
    print("Saved checkpoint:", CHECKPOINT_PATH)
    print("Saved experiment history:", METRICS_PATH)
    print("Checkpoint reload: OK")
    print("Official test set: NOT evaluated")

if __name__ == "__main__":
    main()
