import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


class MulticlassClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(2, 16),
            nn.ReLU(),
            nn.Linear(16, 4),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def multiclass_accuracy(
    logits: torch.Tensor,
    targets: torch.Tensor,
) -> float:
    predictions = torch.argmax(
        logits,
        dim=1,
    )

    accuracy = (
        predictions == targets
    ).float().mean()

    return accuracy.item()


torch.manual_seed(42)

class_0 = (
    torch.randn(50, 2) * 0.4
    + torch.tensor([-2.0, -2.0])
)

class_1 = (
    torch.randn(50, 2) * 0.4
    + torch.tensor([2.0, 2.0])
)

class_2 = (
    torch.randn(50, 2) * 0.4
    + torch.tensor([2.0, -2.0])
)

class_3 = (
    torch.randn(50, 2) * 0.4
    + torch.tensor([-2.0, 2.0])
)

X = torch.cat(
    [
        class_0,
        class_1,
        class_2,
        class_3,
    ],
    dim=0,
)

y = torch.cat(
    [
        torch.zeros(
            50,
            dtype=torch.long,
        ),
        torch.ones(
            50,
            dtype=torch.long,
        ),
        torch.full(
            (50,),
            2,
            dtype=torch.long,
        ),
        torch.full(
            (50,),
            3,
            dtype=torch.long,
        ),
    ],
    dim=0,
)

dataset = TensorDataset(
    X,
    y,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    shuffle=True,
)

device = get_device()

model = MulticlassClassifier().to(
    device
)

loss_function = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

epochs = 100

model.train()

for epoch in range(epochs):
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        logits = model(batch_X)

        loss = loss_function(
            logits,
            batch_y,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


X_device = X.to(device)
y_device = y.to(device)

model.eval()

with torch.inference_mode():
    logits = model(X_device)

accuracy = multiclass_accuracy(
    logits,
    y_device,
)

print(
    "Accuracy:",
    round(accuracy, 4),
)
