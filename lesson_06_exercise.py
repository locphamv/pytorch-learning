import torch
from torch.utils.data import (
    DataLoader,
    TensorDataset,
)


X = torch.rand(
    12,
    2,
)

y = (
    X[:, :1]
    + X[:, 1:]
)


dataset = TensorDataset(
    X,
    y,
)

loader = DataLoader(
    dataset,
    batch_size=5,
)


for batch_X, batch_y in loader:
    print(
        batch_X.shape,
        batch_y.shape,
    )
