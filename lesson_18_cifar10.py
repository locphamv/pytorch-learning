
from pathlib import Path
from collections.abc import Sized

import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets
from torchvision.transforms import v2


BATCH_SIZE = 64
EPOCHS = 5
LEARNING_RATE = 0.001
MODEL_PATH = Path("models/cifar10_cnn.pth")


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_loaders() -> tuple[
    DataLoader, DataLoader, DataLoader, list[str]
]:
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(
            mean=(0.5, 0.5, 0.5),
            std=(0.5, 0.5, 0.5),
        ),
    ])

    full_train = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )

    test_dataset = datasets.CIFAR10(
        root="data",
        train=False,
        download=True,
        transform=transform,
    )

    train_dataset, val_dataset = random_split(
        full_train,
        [45000, 5000],
        generator=torch.Generator().manual_seed(42),
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

    return train_loader, val_loader, test_loader, full_train.classes


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


def show_prediction(
    model: nn.Module,
    loader: DataLoader,
    classes: list[str],
    device: torch.device,
) -> None:
    model.eval()

    images, labels = next(iter(loader))

    with torch.inference_mode():
        logits = model(images[:1].to(device))
        probabilities = torch.softmax(logits[0], dim=0)
        top_probs, top_indices = torch.topk(probabilities, 3)

    actual_class = classes[labels[0].item()]
    predicted_class = classes[top_indices[0].item()]

    prediction_text = "\n".join(
        f"{classes[index.item()]}: {prob.item():.2%}"
        for prob, index in zip(top_probs, top_indices)
    )

    print("\nExample prediction:")
    print("Actual:", actual_class)
    print("Predicted:", predicted_class)
    print(prediction_text)

    image = (images[0] * 0.5 + 0.5).clamp(0, 1)
    image = image.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(image, interpolation="nearest")
    plt.title(
        f"Actual: {actual_class}\n"
        f"Predicted: {predicted_class}\n\n"
        f"{prediction_text}"
    )
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def get_dataset_size(loader: DataLoader) -> int:
    dataset = loader.dataset
    if not isinstance(dataset, Sized):
        raise TypeError("Dataset does not support len()")
    return len(dataset)


def main() -> None:
    torch.manual_seed(42)

    device = get_device()
    print("Using device:", device)

    train_loader, val_loader, test_loader, classes = create_loaders()

    print("Training samples:", get_dataset_size(train_loader))
    print("Validation samples:", get_dataset_size(val_loader))
    print("Test samples:", get_dataset_size(test_loader))

    model = CIFAR10CNN().to(device)
    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    best_val_loss = float("inf")

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
            torch.save(model.state_dict(), MODEL_PATH)

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={val_loss:.4f} "
            f"val_accuracy={val_accuracy:.4f}"
        )

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
    print("Final test accuracy:", round(test_accuracy, 4))

    show_prediction(model, test_loader, classes, device)


if __name__ == "__main__":
    main()
