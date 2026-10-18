
from pathlib import Path
from collections.abc import Sized

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.transforms import v2


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001

AUGMENT_TRAIN = True
RUN_FINAL_TEST = False

MODEL_PATH = Path(
    "models/cifar10_augmented.pth"
    if AUGMENT_TRAIN
    else "models/cifar10_baseline.pth"
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_loaders() -> tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    list[str],
]:
    common_transforms = [
        v2.ToImage(),
        v2.ToDtype(
            torch.float32,
            scale=True,
        ),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ]

    evaluation_transform = v2.Compose(
        common_transforms
    )

    if AUGMENT_TRAIN:
        training_transform = v2.Compose([
            v2.RandomCrop(
                size=32,
                padding=4,
            ),
            v2.RandomHorizontalFlip(
                p=0.5,
            ),
            *common_transforms,
        ])
    else:
        training_transform = evaluation_transform

    # Training dataset uses the selected
    # training transform.
    full_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=training_transform,
    )

    # The same original training images,
    # but without random augmentation.
    full_validation = datasets.CIFAR10(
        root="data",
        train=True,
        download=False,
        transform=evaluation_transform,
    )

    # Official test images always use
    # deterministic preprocessing.
    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=evaluation_transform,
    )

    generator = torch.Generator().manual_seed(
        42
    )

    indices = torch.randperm(
        len(full_train),
        generator=generator,
    ).tolist()

    train_indices = indices[:45000]
    validation_indices = indices[45000:]

    train_dataset = Subset(
        full_train,
        train_indices,
    )

    validation_dataset = Subset(
        full_validation,
        validation_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        generator=(
            torch.Generator().manual_seed(123)
        ),
    )

    val_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        full_train.classes,
    )

class CIFAR10CNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = loss_function(logits, labels)

        optimizer.zero_grad()
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
    correct = 0
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
            correct += (predictions == labels).sum().item()
            total_samples += batch_size

    average_loss = total_loss / total_samples
    accuracy = correct / total_samples

    return average_loss, accuracy

def get_dataset_size(loader: DataLoader) -> int:
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset does not support len()")
    return len(dataset)

def main() -> None:
    torch.manual_seed(42)

    device = get_device()

    print("Using device:", device)
    print("Augmentation enabled:", AUGMENT_TRAIN)

    train_loader, val_loader, test_loader, classes = (
        create_loaders()
    )

    print(
        "Training samples:",
        len(train_loader.dataset),
    )

    print(
        "Validation samples:",
        len(val_loader.dataset),
    )

    print(
        "Test samples:",
        len(test_loader.dataset),
    )

    model = CIFAR10CNN().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    best_val_accuracy = 0.0

    for epoch in range(EPOCHS):
        train_loss = train_one_epoch(
            model,
            train_loader,
            loss_function,
            optimizer,
            device,
        )

        val_loss, val_accuracy = evaluate(
            model,
            val_loader,
            loss_function,
            device,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_accuracy = val_accuracy
            best_epoch = epoch + 1

            torch.save(
                model.state_dict(),
                MODEL_PATH,
            )

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

    print("\nBest validation checkpoint:")
    print("Epoch:", best_epoch)
    print("Loss:", round(best_val_loss, 4))
    print("Accuracy:", round(best_val_accuracy, 4))
    print("Checkpoint:", MODEL_PATH)

    if not RUN_FINAL_TEST:
        print(
            "\nFinal test skipped. "
            "Select the model using validation results."
        )
        return

    # Restore the best validation checkpoint.
    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)
    model.eval()

    # Test the selected model once.
    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print(
        "\nFinal test loss:",
        round(test_loss, 4),
    )

    print(
        "Final test accuracy:",
        round(test_accuracy, 4),
    )

    images, labels = next(iter(test_loader))
    images = images.to(device)

    with torch.inference_mode():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    predicted_class = predictions[0].item()
    actual_class = labels[0].item()

    print("\nExample prediction:")
    print("Actual:", classes[actual_class])
    print("Predicted:", classes[predicted_class])

if __name__ == "__main__":
    main()
