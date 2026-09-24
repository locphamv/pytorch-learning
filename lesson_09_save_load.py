from pathlib import Path

import torch
from torch import nn


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


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def save_model(
    model: nn.Module,
    model_path: Path,
) -> None:
    model_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        model_path,
    )


def load_model(
    model_path: Path,
    device: torch.device,
) -> RegressionModel:
    model = RegressionModel()

    state_dict = torch.load(
        model_path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(state_dict)

    model = model.to(device)
    model.eval()

    return model


device = get_device()

X = torch.tensor([
    [1.0],
    [2.0],
    [3.0],
    [4.0],
    [5.0],
]).to(device)

y = torch.tensor([
    [5.0],
    [8.0],
    [11.0],
    [14.0],
    [17.0],
]).to(device)


model = RegressionModel().to(device)

loss_function = nn.MSELoss()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.01,
)


model.train()

for epoch in range(500):
    predictions = model(X)

    loss = loss_function(
        predictions,
        y,
    )

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()


test_X = torch.tensor([
    [10.0],
]).to(device)

model.eval()

with torch.inference_mode():
    original_prediction = model(test_X)


model_path = Path(
    "models/regression_model.pth"
)

save_model(
    model,
    model_path,
)

loaded_model = load_model(
    model_path,
    device,
)

with torch.inference_mode():
    loaded_prediction = loaded_model(
        test_X
    )


print(
    "Original prediction:",
    original_prediction.item(),
)

print(
    "Loaded prediction:",
    loaded_prediction.item(),
)

print(
    "Predictions match:",
    torch.allclose(
        original_prediction,
        loaded_prediction,
    ),
)
