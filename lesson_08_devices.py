import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
    random_split,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class RegressionModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            in_features=1,
            out_features=1,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.linear(x)


def move_batch_to_device(
    batch_X: torch.Tensor,
    batch_y: torch.Tensor,
    device: torch.device,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    batch_X = batch_X.to(device)
    batch_y = batch_y.to(device)
    return batch_X, batch_y


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        batch_X, batch_y = (
            move_batch_to_device(
                batch_X,
                batch_y,
                device,
            )
        )

        predictions = model(batch_X)

        loss = loss_function(
            predictions,
            batch_y,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_X.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return total_loss / total_samples


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> float:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    with torch.inference_mode():
        for batch_X, batch_y in dataloader:
            batch_X, batch_y = (
                move_batch_to_device(
                    batch_X,
                    batch_y,
                    device,
                )
            )

            predictions = model(batch_X)

            loss = loss_function(
                predictions,
                batch_y,
            )

            batch_size = batch_X.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

    return total_loss / total_samples


device = get_device()

torch.manual_seed(42)

X = torch.linspace(
    -2,
    2,
    120,
).reshape(
    -1,
    1,
)

noise = (
    torch.randn_like(X)
    * 0.3
)

y = (
    3 * X
    + 2
    + noise
)

dataset = TensorDataset(
    X,
    y,
)

(
    train_dataset,
    validation_dataset,
    test_dataset,
) = random_split(
    dataset,
    [84, 18, 18],
    generator=(
        torch.Generator()
        .manual_seed(42)
    ),
)

train_loader = DataLoader(
    train_dataset,
    batch_size=16,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=16,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
)

model = RegressionModel().to(device)

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.05,
)

for epoch in range(100):
    train_one_epoch(
        model,
        train_loader,
        loss_function,
        optimizer,
        device,
    )

    evaluate(
        model,
        validation_loader,
        loss_function,
        device,
    )


test_loss = evaluate(
    model,
    test_loader,
    loss_function,
    device,
)

print(
    "Device:",
    device,
)

print(
    "Final test loss:",
    round(test_loss, 4),
)
