import torch
from torch import nn


class SimpleModel(nn.Module):
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


X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
])

y = torch.tensor([
    [2.0],
    [4.0],
    [6.0],
    [8.0],
])

model = SimpleModel()

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)

model.train()

for epoch in range(200):
    predictions = model(X)

    loss = loss_function(
        predictions,
        y,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


model.eval()

new_x = torch.tensor([
    [5.0],
])

with torch.inference_mode():
    prediction = model(new_x)

print(
    "Prediction for x=5:",
    prediction.item(),
)
