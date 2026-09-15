import torch


scalar = torch.tensor(
    5.0
)

vector = torch.tensor([
    1.0,
    2.0,
    3.0,
])

matrix = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])


print(
    "Vector:",
    vector,
)

print(
    "Shape:",
    vector.shape,
)

print(
    "Dimensions:",
    vector.ndim,
)

print(
    "Dtype:",
    vector.dtype,
)

print(
    "Elements:",
    vector.numel(),
)


random_tensor = torch.rand(
    2,
    3,
)

print(
    "\nRandom tensor:"
)

print(
    random_tensor
)


values = torch.arange(
    1,
    7,
)

reshaped = values.reshape(
    2,
    3,
)

print(
    "\nReshaped:"
)

print(
    reshaped
)


a = torch.tensor([
    [1.0, 2.0],
    [3.0, 4.0],
])

b = torch.tensor([
    [5.0, 6.0],
    [7.0, 8.0],
])

print(
    "\nElement-wise multiplication:"
)

print(
    a * b
)

print(
    "\nMatrix multiplication:"
)

print(
    a @ b
)


if torch.backends.mps.is_available():
    device = torch.device(
        "mps"
    )
else:
    device = torch.device(
        "cpu"
    )

print(
    "\nDevice:",
    device,
)

a = a.to(
    device
)

print(
    "Tensor device:",
    a.device,
)
