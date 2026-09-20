import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


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


X = torch.arange(
    1,
    21,
    dtype=torch.float32,
).reshape(
    -1,
    1,
)

y = 3 * X + 2


dataset = TensorDataset(
    X,
    y,
)

dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)


model = RegressionModel()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.001,
)


model.train()

epochs = 100

for epoch in range(epochs):
    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
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

    average_loss = (
        total_loss
        / total_samples
    )


model.eval()

new_X = torch.tensor([
    [25.0],
])

with torch.inference_mode():
    prediction = model(new_X)


print(
    "Prediction for x=25:",
    prediction.item(),
)
