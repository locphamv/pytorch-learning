"""CV Lesson 8: ResNet18 CIFAR-10 validation error analysis.

Uses the exact 500-image validation subset and pretrained transform from
lessons 20 and 21. No held-out test data is used.

Run:
    python lesson_22_error_analysis.py --checkpoint finetuned
    python lesson_22_error_analysis.py --checkpoint baseline
"""

import argparse
import csv
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.utils import save_image


SEED = 42
VALIDATION_SIZE = 500
DEFAULT_BATCH_SIZE = 16  # Conservative for an 8 GB M1 Mac.
OUTPUT_DIR = Path("outputs")
CHECKPOINTS = {
    "baseline": Path("models/resnet18_cifar10.pth"),
    "finetuned": Path("models/resnet18_cifar10_finetuned.pth"),
}


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def create_validation_loader(batch_size: int) -> tuple[DataLoader, Subset, list[str]]:
    # Match Lesson 6 exactly: same transform, seed, index permutation,
    # and last 500 positions of the 50,000-image official training split.
    dataset = datasets.CIFAR10(
        root="data",
        train=True,
        download=True,
        transform=ResNet18_Weights.DEFAULT.transforms(),
    )
    generator = torch.Generator().manual_seed(SEED)
    indices = torch.randperm(len(dataset), generator=generator).tolist()
    val_dataset = Subset(dataset, indices[-VALIDATION_SIZE:])
    loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    return loader, val_dataset, dataset.classes


def load_model(path: Path, num_classes: int, device: torch.device) -> nn.Module:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing trained checkpoint: {path}. "
            "Run Lesson 6 or 7 first; do not evaluate a random classifier."
        )
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    state_dict = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def collect_predictions(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    model.eval()
    target_batches = []
    predicted_batches = []
    confidence_batches = []
    true_prob_batches = []

    for batch_number, (images, targets) in enumerate(loader):
        images = images.to(device)
        targets = targets.to(device)
        logits = model(images)               # [B, 10]
        probabilities = logits.softmax(1)    # [B, 10]
        confidences, predictions = probabilities.max(1)  # [B], [B]
        true_probabilities = probabilities.gather(
            1, targets.unsqueeze(1)
        ).squeeze(1)

        if batch_number == 0:
            print("First batch tensor shapes:")
            for label, tensor in (
                ("images", images),
                ("targets", targets),
                ("logits", logits),
                ("probabilities", probabilities),
                ("predictions", predictions),
            ):
                print(f"  {label:14s}: {tuple(tensor.shape)}")

        target_batches.append(targets.cpu())
        predicted_batches.append(predictions.cpu())
        confidence_batches.append(confidences.cpu())
        true_prob_batches.append(true_probabilities.cpu())

    return tuple(torch.cat(batches) for batches in (
        target_batches, predicted_batches, confidence_batches, true_prob_batches
    ))


def confusion_matrix(
    targets: torch.Tensor, predictions: torch.Tensor, num_classes: int
) -> torch.Tensor:
    assert targets.shape == predictions.shape
    assert targets.ndim == 1
    # A pair (true=i, predicted=j) maps to the unique bin i*C + j.
    encoded = targets.long() * num_classes + predictions.long()
    return torch.bincount(
        encoded, minlength=num_classes * num_classes
    ).reshape(num_classes, num_classes)


def report_metrics(cm: torch.Tensor, classes: list[str]) -> None:
    num_classes = len(classes)
    print("\nConfusion matrix (rows=true class, columns=prediction):")
    print("     " + "".join(f"{i:>5d}" for i in range(num_classes)))
    for i in range(num_classes):
        counts = "".join(f"{int(n):>5d}" for n in cm[i].tolist())
        print(f"{i:>2d}   {counts}   {classes[i]}")

    support = cm.sum(1)
    predicted_counts = cm.sum(0)
    tp = cm.diag()
    recall = tp.float() / support.clamp(min=1)
    precision = tp.float() / predicted_counts.clamp(min=1)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-12)
    accuracy = tp.sum().item() / cm.sum().item()

    print(f"\nOverall validation accuracy: {accuracy:.2%}")
    print("\nPer-class metrics, weakest recall first:")
    print(f"{'Class':12s} {'Support':>7s} {'Recall':>9s} {'Precision':>10s} {'F1':>8s}")
    for i in torch.argsort(recall).tolist():
        print(
            f"{classes[i]:12s} {support[i].item():7d} "
            f"{recall[i].item():9.2%} {precision[i].item():10.2%} "
            f"{f1[i].item():8.2%}"
        )

    pairs = [
        (int(cm[i, j].item()), i, j)
        for i in range(num_classes)
        for j in range(num_classes)
        if i != j and cm[i, j].item() > 0
    ]
    pairs.sort(reverse=True)
    print("\nMost frequent directed confusions:")
    for count, i, j in pairs[:8]:
        print(f"  {classes[i]:12s} -> {classes[j]:12s}: {count}")


def save_predictions_csv(
    val_dataset: Subset,
    classes: list[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    true_probabilities: torch.Tensor,
    tag: str,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"lesson_22_{tag}_predictions.csv"
    with path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow([
            "validation_index", "original_train_index", "true_class",
            "predicted_class", "pred_confidence", "true_class_probability", "correct"
        ])
        for i in range(len(val_dataset)):
            truth = int(targets[i].item())
            pred = int(predictions[i].item())
            writer.writerow([
                i, int(val_dataset.indices[i]), classes[truth], classes[pred],
                f"{confidences[i].item():.6f}",
                f"{true_probabilities[i].item():.6f}", int(truth == pred),
            ])
    print(f"\nSaved prediction table: {path}")


def save_high_confidence_mistakes(
    val_dataset: Subset,
    classes: list[str],
    targets: torch.Tensor,
    predictions: torch.Tensor,
    confidences: torch.Tensor,
    tag: str,
    max_images: int = 12,
) -> None:
    wrong = (targets != predictions).nonzero(as_tuple=True)[0].tolist()
    wrong.sort(key=lambda i: float(confidences[i]), reverse=True)
    selected = wrong[:max_images]
    if not selected:
        print("No validation mistakes to visualize.")
        return

    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    images = []
    print("\nHighest-confidence mistakes (grid: four columns):")
    for grid_idx, val_idx in enumerate(selected):
        normalized_image, _ = val_dataset[val_idx]
        images.append((normalized_image * std + mean).clamp(0, 1))
        truth = classes[int(targets[val_idx])]
        pred = classes[int(predictions[val_idx])]
        print(
            f"  R{grid_idx // 4 + 1}C{grid_idx % 4 + 1} "
            f"validation #{val_idx}: {truth} -> {pred}; "
            f"confidence={float(confidences[val_idx]):.1%}"
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / f"lesson_22_{tag}_mistakes.png"
    save_image(torch.stack(images), path, nrow=4, padding=4)
    print(f"Saved image grid: {path}")


def print_confusion_asymmetry(cm: torch.Tensor, class_names: list[str], top_k: int = 5) -> None:
    """
    Tính và in ra các cặp nhãn có sự nhầm lẫn bất đối xứng lớn nhất (cm[i, j] != cm[j, i]).
    Ví dụ: Lớp A bị đoán nhầm thành B nhiều hơn hẳn so với B thành A.
    """
    num_classes = cm.shape[0]
    asymmetries = []

    # Duyệt qua các cặp (i, j) với i < j để không lặp lại và bỏ qua đường chéo chính (i == j)
    for i in range(num_classes):
        for j in range(i + 1, num_classes):
            i_to_j = cm[i, j].item()  # Actual i -> Predicted j
            j_to_i = cm[j, i].item()  # Actual j -> Predicted i
            diff = abs(i_to_j - j_to_i)

            if diff > 0:
                asymmetries.append((diff, i, j, i_to_j, j_to_i))

    # Sắp xếp giảm dần theo độ lệch (diff)
    asymmetries.sort(key=lambda x: x[0], reverse=True)

    print("\n--- Top Directional Confusions (Asymmetry) ---")
    if not asymmetries:
        print("Không có nhầm lẫn bất đối xứng nào.")
        return

    for diff, i, j, i_to_j, j_to_i in asymmetries[:top_k]:
        name_i = class_names[i]
        name_j = class_names[j]

        if i_to_j > j_to_i:
            dominant = f"{name_i} → {name_j} ({i_to_j}) nhiều hơn {name_j} → {name_i} ({j_to_i})"
        else:
            dominant = f"{name_j} → {name_i} ({j_to_i}) nhiều hơn {name_i} → {name_j} ({i_to_j})"

        print(f"• {name_i} vs {name_j} | Lệch: {diff} lỗi | Chi tiết: {dominant}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint", choices=list(CHECKPOINTS), default="finetuned"
    )
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    args = parser.parse_args()
    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    # Fail before any dataset download if the requested model is missing.
    path = CHECKPOINTS[args.checkpoint]
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing trained checkpoint: {path}. "
            "Complete Lesson 6 or 7, or choose the other --checkpoint option."
        )

    device = get_device()
    print("Device:", device)
    loader, val_dataset, classes = create_validation_loader(args.batch_size)
    print("Validation samples:", len(val_dataset))
    model = load_model(path, len(classes), device)
    print("Checkpoint:", path)

    targets, predictions, confidences, true_probabilities = collect_predictions(
        model, loader, device
    )
    assert targets.numel() == len(val_dataset)
    cm = confusion_matrix(targets, predictions, len(classes))
    # Gọi hàm phân tích sau khi đã có cm và class_names
    print_confusion_asymmetry(cm, class_names=val_dataset.dataset.classes)
    assert int(cm.sum()) == len(val_dataset)
    report_metrics(cm, classes)

    correct = predictions == targets
    for name, mask in (("correct", correct), ("incorrect", ~correct)):
        if mask.any():
            print(
                f"Mean confidence ({name}): {float(confidences[mask].mean()):.2%}")

    save_predictions_csv(
        val_dataset, classes, targets, predictions,
        confidences, true_probabilities, args.checkpoint
    )
    save_high_confidence_mistakes(
        val_dataset, classes, targets, predictions, confidences, args.checkpoint
    )
    print("\nHeld-out CIFAR-10 test set was not accessed.")


if __name__ == "__main__":
    main()
