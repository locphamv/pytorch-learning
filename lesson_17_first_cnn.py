import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)


class SmallCNN(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(8, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 8 * 8, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def create_toy_dataset() -> TensorDataset:
    images = torch.zeros(
        90, 3, 32, 32
    )

    labels = torch.arange(
        3
    ).repeat_interleave(30)

    # Class 0: vertical bar
    images[:30, :, :, 14:18] = 1.0

    # Class 1: horizontal bar
    images[30:60, :, 14:18, :] = 1.0

    # Class 2: center square
    images[60:, :, 11:21, 11:21] = 1.0

    # Add small random noise.
    noise = torch.rand_like(images) * 0.15

    images = (
        images + noise
    ).clamp(0, 1)

    return TensorDataset(
        images,
        labels,
    )


def evaluate_accuracy(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> float:
    model.eval()

    correct = 0
    total = 0

    with torch.inference_mode():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            predictions = logits.argmax(
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

    return correct / total


def main() -> None:
    torch.manual_seed(42)

    device = get_device()

    print("Device:", device)

    dataset = create_toy_dataset()

    train_dataset, val_dataset, test_dataset = (
        random_split(
            dataset,
            [60, 15, 15],
            generator=(
                torch.Generator()
                .manual_seed(42)
            ),
        )
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=12,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=12,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=12,
    )

    model = SmallCNN().to(device)

    loss_function = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.003,
    )

    epochs = 10

    for epoch in range(epochs):
        model.train()

        total_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)

            loss = loss_function(
                logits,
                labels,
            )

            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

        average_loss = (
            total_loss
            / len(train_dataset)
        )

        val_accuracy = evaluate_accuracy(
            model,
            val_loader,
            device,
        )

        print(
            f"Epoch {epoch + 1:02d} "
            f"loss={average_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )

    test_accuracy = evaluate_accuracy(
        model,
        test_loader,
        device,
    )

    print(
        "\nFinal test accuracy:",
        round(test_accuracy, 4),
    )

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    print(
        "Trainable parameters:",
        total_parameters,
    )


if __name__ == "__main__":
    main()
