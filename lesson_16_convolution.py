import torch
from torch import nn
from torch.nn import functional as F


def manual_convolution(
    image: torch.Tensor,
    kernel: torch.Tensor,
) -> torch.Tensor:
    image_height, image_width = image.shape
    kernel_height, kernel_width = kernel.shape

    output_height = (
        image_height - kernel_height + 1
    )

    output_width = (
        image_width - kernel_width + 1
    )

    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            "Kernel cannot be larger than the image"
        )

    output = torch.zeros(
        output_height,
        output_width,
        dtype=image.dtype,
        device=image.device,
    )

    for row in range(output_height):
        for col in range(output_width):
            patch = image[
                row:row + kernel_height,
                col:col + kernel_width,
            ]

            output[row, col] = (
                patch * kernel
            ).sum()

    return output


def main() -> None:
    image = torch.tensor([
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
        [0., 0., 0., 1., 1.],
    ])

    kernel = torch.tensor([
        [-1., 0., 1.],
        [-1., 0., 1.],
        [-1., 0., 1.],
    ])

    manual_output = manual_convolution(
        image,
        kernel,
    )

    print("Manual convolution:")
    print(manual_output)

    image_4d = image.unsqueeze(0).unsqueeze(0)
    kernel_4d = kernel.unsqueeze(0).unsqueeze(0)

    pytorch_output = F.conv2d(
        image_4d,
        kernel_4d,
        bias=None,
        stride=1,
        padding=0,
    )

    print(
        "\nPyTorch output:",
        pytorch_output[0, 0],
    )

    assert torch.allclose(
        manual_output,
        pytorch_output[0, 0],
    )

    print("\nManual and PyTorch results match.")

    images = torch.rand(
        4, 3, 64, 64
    )

    conv = nn.Conv2d(
        in_channels=3,
        out_channels=8,
        kernel_size=3,
        stride=1,
        padding=1,
    )

    features = conv(images)

    print("\nRGB input shape:", images.shape)
    print("Feature maps shape:", features.shape)
    print("Kernel shape:", conv.weight.shape)

    parameter_count = sum(
        p.numel()
        for p in conv.parameters()
    )

    print("Parameters:", parameter_count)

    loss = features.square().mean()
    loss.backward()

    print(
        "Gradient shape:",
        conv.weight.grad.shape,
    )


if __name__ == "__main__":
    main()


