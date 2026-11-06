"""CV Lesson 14: NMS, precision/recall, and simplified detection AP/mAP.

Run:
    python lesson_28_nms_precision_map.py

Dependencies: torch, torchvision (matching installed versions).
This educational evaluator is NOT an official COCO evaluator.
"""
from __future__ import annotations

from typing import TypedDict

import torch
from torchvision.ops import batched_nms, box_iou, nms


class Detections(TypedDict):
    boxes: torch.Tensor   # [N, 4] in absolute xyxy coordinates
    labels: torch.Tensor  # [N] integer class IDs
    scores: torch.Tensor  # [N], confidence scores


class GroundTruth(TypedDict):
    boxes: torch.Tensor   # [N, 4]
    labels: torch.Tensor  # [N]


def validate_xyxy(boxes: torch.Tensor) -> None:
    if boxes.ndim != 2 or boxes.shape[1] != 4:
        raise ValueError("boxes must have shape [N, 4]")
    if not torch.is_floating_point(boxes):
        raise ValueError("box coordinates must be floating point")
    if not bool(torch.isfinite(boxes).all()):
        raise ValueError("box coordinates must be finite")
    if bool((boxes[:, 2:] <= boxes[:, :2]).any()):
        raise ValueError("all boxes must have positive area")


def iou_with_one(box: torch.Tensor, boxes: torch.Tensor) -> torch.Tensor:
    """Pairwise IoU of a single xyxy box with boxes [N,4] -> [N]."""
    top_left = torch.maximum(box[:2], boxes[:, :2])
    bottom_right = torch.minimum(box[2:], boxes[:, 2:])
    intersection = (bottom_right - top_left).clamp(min=0).prod(dim=1)
    area_box = (box[2:] - box[:2]).prod()
    areas = (boxes[:, 2:] - boxes[:, :2]).prod(dim=1)
    union = area_box + areas - intersection
    return intersection / union  # positive-area boxes -> positive union


def nms_from_scratch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Greedy class-agnostic NMS; returns original indices by score."""
    validate_xyxy(boxes)
    if scores.shape != (boxes.shape[0],):
        raise ValueError("scores must be [N]")
    if not 0 <= iou_threshold <= 1:
        raise ValueError("iou_threshold must be between 0 and 1")
    if not bool(torch.isfinite(scores).all()):
        raise ValueError("scores must be finite")

    order = torch.argsort(scores, descending=True, stable=True)
    kept: list[int] = []
    while order.numel():
        best = order[0]
        kept.append(int(best.item()))
        remaining = order[1:]
        if not remaining.numel():
            break
        overlaps = iou_with_one(boxes[best], boxes[remaining])
        # Suppress IoU strictly GREATER than threshold (Torchvision convention).
        order = remaining[overlaps <= iou_threshold]
    return torch.tensor(kept, dtype=torch.long, device=boxes.device)


def class_aware_nms_from_scratch(
    boxes: torch.Tensor,
    scores: torch.Tensor,
    labels: torch.Tensor,
    iou_threshold: float,
) -> torch.Tensor:
    """Perform NMS independently for each class; sort kept indices by score."""
    validate_xyxy(boxes)
    if labels.shape != (boxes.shape[0],) or scores.shape != (boxes.shape[0],):
        raise ValueError("labels and scores must both be [N]")
    kept_parts = []
    for class_id in torch.unique(labels):
        original_indices = torch.where(labels == class_id)[0]
        local_kept = nms_from_scratch(
            boxes[original_indices], scores[original_indices], iou_threshold
        )
        kept_parts.append(original_indices[local_kept])
    if not kept_parts:
        return torch.empty(0, dtype=torch.long, device=boxes.device)
    kept = torch.cat(kept_parts)
    return kept[torch.argsort(scores[kept], descending=True, stable=True)]


def apply_class_aware_nms(
    detections: Detections, iou_threshold: float = 0.5
) -> Detections:
    keep = class_aware_nms_from_scratch(
        detections["boxes"], detections["scores"], detections["labels"],
        iou_threshold,
    )
    return {key: values[keep] for key, values in detections.items()}  # type: ignore[return-value]


def evaluate_one_class(
    ground_truth: list[GroundTruth],
    predictions: list[Detections],
    class_id: int,
    match_iou_threshold: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Score-sort all detections; greedy match one GT at most once per image.

    Returns (scores, tp_flags, fp_flags, recall, total_gt). All arrays are
    sorted by descending confidence. No GT use by a different class allowed.
    """
    if len(ground_truth) != len(predictions):
        raise ValueError("One prediction dict is required for each image")
    if not 0 <= match_iou_threshold <= 1:
        raise ValueError("match_iou_threshold must be in [0, 1]")

    class_boxes = [g["boxes"][g["labels"] == class_id] for g in ground_truth]
    matched = [torch.zeros(len(boxes), dtype=torch.bool) for boxes in class_boxes]
    total_gt = sum(len(boxes) for boxes in class_boxes)

    candidates: list[tuple[float, int, torch.Tensor]] = []
    for image_id, detections in enumerate(predictions):
        which = torch.where(detections["labels"] == class_id)[0]
        for idx in which.tolist():
            candidates.append((float(detections["scores"][idx]), image_id,
                               detections["boxes"][idx]))
    # Python's stable sort preserves input order for exact ties.
    candidates.sort(key=lambda row: row[0], reverse=True)

    scores_list: list[float] = []
    tp_list: list[int] = []
    fp_list: list[int] = []
    for score, image_id, box in candidates:
        scores_list.append(score)
        gt_boxes = class_boxes[image_id]
        is_tp = False
        if gt_boxes.numel():
            overlaps = iou_with_one(box, gt_boxes)
            # Only unmatched boxes are eligible. A GT can be matched once.
            overlaps[matched[image_id]] = -1.0
            best_iou, best_index = overlaps.max(dim=0)
            if best_iou.item() >= match_iou_threshold:
                matched[image_id][best_index] = True
                is_tp = True
        tp_list.append(int(is_tp))
        fp_list.append(int(not is_tp))

    scores_tensor = torch.tensor(scores_list, dtype=torch.float64)
    tp = torch.tensor(tp_list, dtype=torch.float64)
    fp = torch.tensor(fp_list, dtype=torch.float64)
    cumulative_tp = tp.cumsum(0)
    recall = cumulative_tp / total_gt if total_gt else torch.zeros_like(tp)
    return scores_tensor, tp, fp, recall, total_gt


def precision_recall_ap101(
    tp: torch.Tensor, fp: torch.Tensor, total_gt: int
) -> tuple[torch.Tensor, torch.Tensor, float]:
    """COCO-style 101-point interpolated AP for ONE class and ONE IoU threshold.

    Simplification: no crowd/ignore regions, area ranges, or max-detections rules.
    Class must have at least one ground-truth instance.
    """
    if total_gt <= 0:
        raise ValueError("AP is undefined for a class without ground truth")
    accumulated_tp = tp.cumsum(0)
    accumulated_fp = fp.cumsum(0)
    precision = accumulated_tp / (accumulated_tp + accumulated_fp).clamp(min=1)
    recall = accumulated_tp / total_gt
    recall_samples = torch.linspace(0, 1, 101, dtype=torch.float64)
    interpolated = []
    for target in recall_samples:
        eligible = precision[recall >= target]
        interpolated.append(float(eligible.max()) if eligible.numel() else 0.0)
    ap = sum(interpolated) / len(interpolated)
    return precision, recall, ap


def make_demo_data() -> tuple[list[GroundTruth], list[Detections]]:
    """Two images, two classes: cat=0, dog=1. Fully synthetic coordinates."""
    gt = [
        {"boxes": torch.tensor([[0., 0., 40., 40.],
                                 [60., 0., 100., 40.]]),
         "labels": torch.tensor([0, 1])},
        {"boxes": torch.tensor([[10., 10., 50., 50.]]),
         "labels": torch.tensor([0])},
    ]
    raw = [
        {"boxes": torch.tensor([[0., 0., 40., 40.],
                                 [2., 2., 42., 42.],
                                 [60., 0., 100., 40.],
                                 [110., 110., 130., 130.]]),
         "labels": torch.tensor([0, 0, 1, 0]),
         "scores": torch.tensor([0.95, 0.85, 0.90, 0.50])},
        {"boxes": torch.tensor([[65., 65., 90., 90.],
                                 [14., 14., 54., 54.],
                                 [15., 15., 55., 55.],
                                 [10., 10., 50., 50.]]),
         "labels": torch.tensor([0, 0, 0, 1]),
         "scores": torch.tensor([0.88, 0.70, 0.65, 0.75])},
    ]
    return gt, raw


def main() -> None:
    torch.set_printoptions(precision=4, sci_mode=False)
    # NMS verification: exact duplicate geometry, DIFFERENT category.
    boxes = torch.tensor([
        [10., 10., 50., 50.],
        [12., 12., 52., 52.],
        [60., 10., 100., 50.],
        [10., 10., 50., 50.],
    ])
    scores = torch.tensor([0.95, 0.80, 0.70, 0.90])
    labels = torch.tensor([0, 0, 0, 1])
    assert torch.isclose(iou_with_one(boxes[0], boxes[1:2])[0],
                         box_iou(boxes[0:1], boxes[1:2])[0, 0])
    kept_global = nms_from_scratch(boxes, scores, iou_threshold=0.5)
    kept_by_class = class_aware_nms_from_scratch(boxes, scores, labels, 0.5)
    assert kept_global.tolist() == nms(boxes, scores, 0.5).tolist() == [0, 2]
    assert kept_by_class.tolist() == batched_nms(boxes, scores, labels, 0.5).tolist() == [0, 3, 2]
    print("GLOBAL NMS kept indices:", kept_global.tolist())
    print("CLASS-AWARE NMS kept:", kept_by_class.tolist())

    ground_truth, raw_predictions = make_demo_data()
    predictions = [apply_class_aware_nms(p, 0.5) for p in raw_predictions]
    print("\nDETECTIONS BEFORE / AFTER NMS:",
          [(len(r["scores"]), len(p["scores"]))
           for r, p in zip(raw_predictions, predictions)])

    for class_id, name in [(0, "cat"), (1, "dog")]:
        scores_sorted, tp, fp, _, total = evaluate_one_class(
            ground_truth, predictions, class_id, match_iou_threshold=0.5
        )
        precision, recall, ap = precision_recall_ap101(tp, fp, total)
        print(f"\n{name.upper()} @ IoU >= 0.50: GT={total}, AP101={ap:.6f}")
        print("  score   TP FP precision recall")
        for i in range(len(tp)):
            print(f"  {scores_sorted[i]:.2f}     {int(tp[i])}  {int(fp[i])}"
                  f"   {precision[i]:.3f}    {recall[i]:.3f}")
        if class_id == 0:
            assert tp.tolist() == [1.0, 0.0, 1.0, 0.0]
            assert abs(ap - (51 + 50 * (2/3)) / 101) < 1e-6
        else:
            assert tp.tolist() == [1.0, 0.0]
            assert abs(ap - 1.0) < 1e-12

    def mean_ap(iou_threshold: float) -> float:
        per_class = []
        for class_id in (0, 1):
            _, tp, fp, _, total = evaluate_one_class(
                ground_truth, predictions, class_id, iou_threshold
            )
            _, _, ap = precision_recall_ap101(tp, fp, total)
            per_class.append(ap)
        return sum(per_class) / len(per_class)

    ap50 = mean_ap(0.50)
    ap75 = mean_ap(0.75)
    ap50_95 = sum(mean_ap(i / 100) for i in range(50, 100, 5)) / 10
    print(f"\nSimplified mAP@0.50: {ap50:.6f}")
    print(f"Simplified mAP@0.75: {ap75:.6f}")
    print(f"Simplified mAP@0.50:0.95: {ap50_95:.6f}")
    assert ap75 < ap50
    print("All checks passed. This is not an official COCO evaluator.")


if __name__ == "__main__":
    main()
