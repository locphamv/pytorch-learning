import torch
from torch import nn


class DeepModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(3, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
            nn.Linear(4, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


X = torch.rand(
    10,
    3,
)

model = DeepModel()

output = model(X)

total_parameters = sum(
    parameter.numel()
    for parameter in model.parameters()
)

print("Output shape:", output.shape)
print("Parameters:", total_parameters)
