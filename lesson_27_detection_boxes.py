
import torch
from torchvision.ops import box_convert, box_iou


def validate_xyxy(boxes: torch.Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError(
            "Expected boxes of shape [N, 4]"
        )

    if not torch.is_floating_point(boxes):
        raise ValueError(
            "Use floating-point box coordinates"
        )

    if not bool(torch.isfinite(boxes).all()):
        raise ValueError(
            "Box coordinates must be finite"
        )

    if bool((boxes[:, 2:] < boxes[:, :2]).any()):
        raise ValueError(
            "Invalid box corner ordering"
        )


def pairwise_iou(
    boxes_a: torch.Tensor,
    boxes_b: torch.Tensor,
) -> torch.Tensor:
    validate_xyxy(boxes_a)
    validate_xyxy(boxes_b)

    # Shape: [N, M, 2]
    top_left = torch.maximum(
        boxes_a[:, None, :2],
        boxes_b[None, :, :2],
    )

    bottom_right = torch.minimum(
        boxes_a[:, None, 2:],
        boxes_b[None, :, 2:],
    )

    # Shape: [N, M, 2]
    intersection_wh = (
        bottom_right - top_left
    ).clamp(min=0)

    # Shape: [N, M]
    intersection = intersection_wh.prod(dim=-1)

    # Shapes: [N] and [M]
    area_a = (
        boxes_a[:, 2:] - boxes_a[:, :2]
    ).prod(dim=-1)

    area_b = (
        boxes_b[:, 2:] - boxes_b[:, :2]
    ).prod(dim=-1)

    # Shape: [N, M]
    union = (
        area_a[:, None]
        + area_b[None, :]
        - intersection
    )

    # Treat zero-union degenerate boxes as IoU 0.
    return torch.where(
        union > 0,
        intersection / union.clamp(min=1e-12),
        torch.zeros_like(union),
    )

def xywh_to_xyxy(boxes: torch.tensor) -> torch.Tensor:
    """Chuyển đổi tensor bounding boxes từ [x1, y1, w, h] sang [x1, y1, x2,
    y2]."""
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("Expected boxes of shape [N, 4]")
    x1,y1,w,h = boxes.unbind(dim=-1)
    x2 = x1 + w
    y2 = y1 + h
    return torch.stack([x1,y1,x2,y2], dim=-1)


def main() -> None:
    ground_truth = torch.tensor([
        [10., 10., 50., 50.],
        [0., 0., 20., 20.],
    ])

    predictions = torch.tensor([
        [30., 20., 70., 60.],
        [10., 10., 50., 50.],
        [72., 55., 78., 65.],
    ])

    ious = pairwise_iou(
        ground_truth,
        predictions,
    )

    reference = box_iou(
        ground_truth,
        predictions,
    )

    # Verify against Torchvision.
    torch.testing.assert_close(ious, reference)

    print("Ground truth:", ground_truth.shape)
    print("Predictions:", predictions.shape)
    print("IoU matrix:", ious.shape)
    print(ious)

    # Verify the worked example.
    assert abs(
        ious[0, 0].item() - 600 / 2600
    ) < 1e-6

    assert ious[0, 1].item() == 1.0
    assert ious[0, 2].item() == 0.0

    xywh = box_convert(
        ground_truth,
        in_fmt="xyxy",
        out_fmt="xywh",
    )

    cxcywh = box_convert(
        ground_truth,
        in_fmt="xyxy",
        out_fmt="cxcywh",
    )

    print("XYWH:\n", xywh)
    print("CXCYWH:\n", cxcywh)

    # Normalize one box relative to an 80x70 image.
    scale = torch.tensor([80., 70., 80., 70.])
    normalized = ground_truth[0] / scale

    print("Normalized first box:", normalized)
    print("All checks passed.")
    print("\n--- RUNNING EXERCISES ---")

    # 1. Exercise A: Tính IoU bằng tay (400 / 2800 = 1/7)
    box_a = torch.tensor([[0.0, 0.0, 40.0, 40.0]])
    box_b = torch.tensor([[20.0, 20.0, 60.0, 60.0]])
    iou_ex_a = pairwise_iou(box_a, box_b).item()
    print("Exercise A IoU:", round(iou_ex_a, 6))
    assert abs(iou_ex_a - (400.0 / 2800.0)) < 1e-6

    # 2. Exercise B: Kiểm tra rời nhau (0.0) và trùng khít (1.0)
    disjoint_box = torch.tensor([[100.0, 100.0, 140.0, 140.0]])
    iou_disjoint = pairwise_iou(box_a, disjoint_box).item()
    iou_identical = pairwise_iou(box_a, box_a).item()
    print(
        f"Exercise B - Disjoint IoU: {iou_disjoint}, Identical IoU: {iou_identical}"
    )
    assert iou_disjoint == 0.0
    assert iou_identical == 1.0

    # 3. Exercise C: Kiểm tra hàm xywh_to_xyxy tự viết với torchvision
    ex_c_boxes = torch.tensor(
        [
            [10.0, 20.0, 40.0, 30.0],
            [0.0, 0.0, 20.0, 20.0],
        ]
    )
    custom_xyxy = xywh_to_xyxy(ex_c_boxes)
    expected_xyxy = box_convert(ex_c_boxes, in_fmt="xywh", out_fmt="xyxy")
    torch.testing.assert_close(custom_xyxy, expected_xyxy)
    print("Exercise C xywh -> xyxy:\n", custom_xyxy)
    print("Exercise C assertion: Passed!")

    print("\nAll checks passed.")

if __name__ == "__main__":
    main()
