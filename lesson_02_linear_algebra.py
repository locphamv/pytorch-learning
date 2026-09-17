import torch


X = torch.tensor([
    [6.0, 1.0, 7.5],
    [2.0, 5.0, 4.0],
    [8.0, 0.0, 9.0],
    [4.0, 3.0, 6.0],
])


W = torch.tensor([
    [0.5, 0.2],
    [-0.3, 0.7],
    [0.8, -0.4],
])


b = torch.tensor([
    0.2,
    -0.1,
])


print(
    "Input shape:",
    X.shape,
)

print(
    "Weight shape:",
    W.shape,
)

print(
    "Bias shape:",
    b.shape,
)


weighted_output = (
    X @ W
)

print(
    "\nBefore bias:"
)

print(
    weighted_output
)

print(
    "Shape:",
    weighted_output.shape,
)


output = (
    weighted_output
    + b
)

print(
    "\nAfter bias:"
)

print(
    output
)

print(
    "Output shape:",
    output.shape,
)


x = torch.arange(
    12
)

print(
    "\nOriginal:",
    x.shape,
)

reshaped = x.reshape(
    3,
    -1,
)

print(
    "Reshaped:",
    reshaped.shape,
)


single_sample = torch.tensor([
    6.0,
    1.0,
    7.5,
])

print(
    "\nSingle sample:",
    single_sample.shape,
)

batched_sample = (
    single_sample.unsqueeze(
        0
    )
)

print(
    "Batched sample:",
    batched_sample.shape,
)
