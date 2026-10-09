from pathlib import Path

import torch
from torchvision.io import decode_image
from torchvision.transforms import v2


image_path = Path("data/sample.png")
image = decode_image(str(image_path))

print("Original shape:", image.shape)
print("Original dtype:", image.dtype)

red_channel = image[0]
pixel = image[:, 0, 0]

print("Red channel shape:", red_channel.shape)
print("Top-left RGB pixel:", pixel)

transform = v2.Compose([
    v2.Resize(
        (128, 128),
        antialias=True,
    ),
    v2.ToDtype(
        torch.float32,
        scale=True,
    ),
])

processed_image = transform(image)
batch = processed_image.unsqueeze(0)

print("Processed shape:", processed_image.shape)
print("Processed dtype:", processed_image.dtype)
print("Processed range:", processed_image.min().item(), processed_image.max().item())
print("Batch shape:", batch.shape)
print("Values per image:", processed_image.numel())
