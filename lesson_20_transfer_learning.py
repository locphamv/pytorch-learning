
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import (
    ResNet18_Weights,
    resnet18,
)


BATCH_SIZE = 32
EPOCHS = 3
LEARNING_RATE = 0.001

TRAIN_SAMPLES = 2000
VAL_SAMPLES = 500
TEST_SAMPLES = 1000

RUN_FINAL_TEST = False

MODEL_PATH = Path(
    "models/resnet18_cifar10.pth"
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
    weights = ResNet18_Weights.DEFAULT
    transform = weights.transforms()

    full_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    full_test = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(42)

    indices = torch.randperm(
        len(full_train),
        generator=generator,
    ).tolist()

    train_indices = indices[:TRAIN_SAMPLES]
    val_indices = indices[-VAL_SAMPLES:]

    train_dataset = Subset(
        full_train,
        train_indices,
    )

    val_dataset = Subset(
        full_train,
        val_indices,
    )

    test_dataset = Subset(
        full_test,
        list(range(TEST_SAMPLES)),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
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


def create_model(
    num_classes: int,
) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT

    model = resnet18(
        weights=weights,
    )

    # Freeze the pretrained feature extractor.
    for parameter in model.parameters():
        parameter.requires_grad = False

    input_features = model.fc.in_features

    # Replace the ImageNet classification head.
    model.fc = nn.Linear(
        input_features,
        num_classes,
    )

    return model


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    # Keep the frozen backbone's BatchNorm
    # statistics unchanged.
    model.eval()

    # The new classifier remains trainable.
    model.fc.train()

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
            correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_size

    return (
        total_loss / total_samples,
        correct / total_samples,
    )


def main() -> None:
    torch.manual_seed(42)

    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, test_loader, classes = (
        create_loaders()
    )

    print("Training samples:", len(train_loader.dataset))
    print("Validation samples:", len(val_loader.dataset))
    print("Test samples:", len(test_loader.dataset))

    model = create_model(
        num_classes=len(classes),
    ).to(device)

    trainable_parameters = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    print(
        "Trainable parameters:",
        trainable_parameters,
    )

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.fc.parameters(),
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
            f"val_acc={val_accuracy:.4f}"
        )

    print("\nBest validation checkpoint:")
    print("Epoch:", best_epoch)
    print("Loss:", round(best_val_loss, 4))
    print("Accuracy:", round(best_val_accuracy, 4))

    if not RUN_FINAL_TEST:
        print("\nFinal test skipped.")
        return

    state_dict = torch.load(
        MODEL_PATH,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    test_loss, test_accuracy = evaluate(
        model,
        test_loader,
        loss_function,
        device,
    )

    print("\nFinal test loss:", round(test_loss, 4))
    print(
        "Final test accuracy:",
        round(test_accuracy, 4),
    )

    images, labels = next(iter(test_loader))
    images = images.to(device)

    model.eval()

    with torch.inference_mode():
        logits = model(images)
        predictions = logits.argmax(dim=1)

    print("\nExample prediction:")
    print("Actual:", classes[labels[0].item()])
    print(
        "Predicted:",
        classes[predictions[0].item()],
    )


if __name__ == "__main__":
    main()
