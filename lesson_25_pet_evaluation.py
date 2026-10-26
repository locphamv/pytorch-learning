"""Validate pet classifiers, lock a selection, then evaluate the held-out test set."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms.functional import to_pil_image

from lesson_23_pet_capstone_baseline import get_device

DATA_DIR = Path("data")
RESULTS_ROOT = Path("runs/pet_breeds_evaluation")
SELECTION_PATH = RESULTS_ROOT / "selected_model.json"
FINAL_DIR = RESULTS_ROOT / "official_test"
BATCH_SIZE = 8
PREPROCESSING_ID = "MobileNet_V3_Small_Weights.DEFAULT.transforms()"

def file_sha256(path: Path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_checkpoint(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing trained checkpoint: {path}")
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    required = {
        "checkpoint_version", "architecture", "num_classes", "class_names",
        "preprocessing", "train_indices", "val_indices", "model_state_dict",
        "validation_loss", "validation_accuracy",
    }
    if not isinstance(checkpoint, dict) or not required.issubset(checkpoint):
        raise ValueError("Unexpected checkpoint format; expected Lesson 23/24 artifact")
    if checkpoint["checkpoint_version"] != 1:
        raise ValueError("Unsupported checkpoint version")
    if checkpoint["architecture"] != "mobilenet_v3_small":
        raise ValueError("Expected mobilenet_v3_small architecture")
    if checkpoint["preprocessing"] != PREPROCESSING_ID:
        raise ValueError("Checkpoint preprocessing differs from training")
    if not isinstance(checkpoint["class_names"], list) or len(checkpoint["class_names"]) != checkpoint["num_classes"]:
        raise ValueError("Invalid class metadata")
    if checkpoint["num_classes"] != 37:
        raise ValueError("Expected the 37-class Oxford-IIIT Pet dataset")
    train_idx = checkpoint["train_indices"]
    val_idx = checkpoint["val_indices"]
    for indices in (train_idx, val_idx):
        if not isinstance(indices, list) or any(type(i) is not int or i < 0 for i in indices):
            raise ValueError("Split indices must be lists of nonnegative integers")
    if not train_idx:
        raise ValueError("Empty training set")
    if len(train_idx) != len(set(train_idx)) or len(val_idx) != len(set(val_idx)):
        raise ValueError("Duplicate indices in saved split")
    if set(train_idx) & set(val_idx):
        raise ValueError("Train and validation indices overlap")
    if len(val_idx) == 0:
        raise ValueError("Empty validation set")
    return checkpoint


def build_model(checkpoint: dict, device: torch.device):
    model = mobilenet_v3_small(weights=None)
    final_in = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(final_in, checkpoint["num_classes"])
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model = model.to(device)
    model.eval()
    return model

def load_data(checkpoint: dict, split: str):
    if split not in ("validation", "test"):
        raise ValueError("Invalid evaluation split")
    source = OxfordIIITPet(
        root=DATA_DIR,
        split="trainval" if split == "validation" else "test",
        target_types="category",
        download=True,
        transform=MobileNet_V3_Small_Weights.DEFAULT.transforms()
    )
    if list(source.classes) != checkpoint["class_names"]:
        raise ValueError("Breed/class order differs from saved checkpoint")
    if split == "validation":
        train_idx = checkpoint["train_indices"]
        val_idx = checkpoint["val_indices"]
        if (set(train_idx) | set(val_idx)) != set(range(len(source))):
            raise ValueError("Saved split does not cover the current trainval dataset")
        dataset = Subset(source, val_idx)
        source_indices = val_idx
    else:
        dataset = source
        source_indices = list(range(len(source)))
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False,  num_workers=0)
    return dataset, loader, source_indices


def validation_dir(path: Path) -> Path:
    # Include checkpoint content so candidates in the same folder cannot collide.
    return RESULTS_ROOT / f"{path.stem}-{file_sha256(path)}" / "validation"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def save_error_grid(dataset, rows: list[dict], path: Path) -> None:
    errors = sorted((row for row in rows if not row["correct"]),
                    key=lambda row: row["confidence"], reverse=True)[:12]
    columns, width, height = 3, 280, 290
    canvas = Image.new("RGB", (columns * width, max(1, (len(errors) + 2) // 3) * height), "white")
    draw = ImageDraw.Draw(canvas)
    if not errors:
        draw.text((12, 12), "No misclassified examples.", fill="black")
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    mean = torch.tensor(transform.mean).view(3, 1, 1)
    std = torch.tensor(transform.std).view(3, 1, 1)
    for i, row in enumerate(errors):
        tensor, _ = dataset[row["dataset_index"]]
        picture = to_pil_image((tensor.cpu() * std + mean).clamp(0, 1))
        picture.thumbnail((width, 224))
        x, y = (i % columns) * width, (i // columns) * height
        canvas.paste(picture, (x, y))
        draw.text((x + 5, y + 228),
                  f"True: {row['true_breed']}\nPred: {row['pred_breed']}\nConfidence: {row['confidence']:.1%}",
                  fill="black")
    canvas.save(path)


def evaluate_checkpoint(path: Path, split: str, output_dir: Path) -> dict:
    checkpoint = load_checkpoint(path)
    device = get_device()
    model = build_model(checkpoint, device)
    dataset, loader, source_indices = load_data(checkpoint, split)
    class_names = checkpoint["class_names"]
    targets_batches, probability_batches = [], []
    total_loss = 0.0
    with torch.inference_mode():
        for images, labels in loader:
            labels = labels.to(device)
            logits = model(images.to(device))
            total_loss += nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            targets_batches.append(labels.cpu())
            probability_batches.append(logits.softmax(dim=1).cpu())
    if not targets_batches:
        raise ValueError("Cannot evaluate an empty dataset")
    targets = torch.cat(targets_batches)
    probabilities = torch.cat(probability_batches)
    confidence, predictions = probabilities.max(dim=1)
    true_prob = probabilities[torch.arange(len(targets)), targets]
    loss = total_loss / len(targets)
    overall_acc = (predictions == targets).float().mean().item()
    n = len(class_names)
    cm = torch.bincount(targets * n + predictions, minlength=n * n).reshape(n, n)
    per_class = []
    for i, breed in enumerate(class_names):
        tp = int(cm[i, i])
        support, predicted = int(cm[i].sum()), int(cm[:, i].sum())
        precision = tp / predicted if predicted else 0.0
        recall = tp / support if support else 0.0
        f1 = 2 * tp / (support + predicted) if support + predicted else 0.0
        per_class.append(dict(breed=breed, support=support, precision=precision, recall=recall, f1=f1))
    macro_f1 = sum(row["f1"] for row in per_class) / n
    mistakes = sorted([
        {"true": class_names[i], "predicted": class_names[j], "count": int(cm[i, j])}
        for i in range(n) for j in range(n) if i != j and cm[i, j] > 0
    ], key=lambda row: row["count"], reverse=True)
    all_rows = []
    for i, (t, p) in enumerate(zip(targets.tolist(), predictions.tolist())):
        all_rows.append({
            "dataset_index": i, "source_index": source_indices[i],
            "true_id": t, "true_breed": class_names[t],
            "pred_id": p, "pred_breed": class_names[p],
            "confidence": round(float(confidence[i]), 7),
            "true_class_probability": round(float(true_prob[i]), 7),
            "correct": int(t == p),
        })
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "predictions.csv", all_rows, list(all_rows[0]))
    write_csv(output_dir / "per_class.csv", per_class, list(per_class[0]))
    write_csv(output_dir / "confusion_matrix.csv", [
        {"actual_breed": name, **{breed: int(cm[i, j]) for j, breed in enumerate(class_names)}}
        for i, name in enumerate(class_names)
    ], ["actual_breed", *class_names])
    save_error_grid(dataset, all_rows, output_dir / "high_confidence_mistakes.png")
    summary = {
        "split": split, "checkpoint": str(path.resolve()),
        "checkpoint_sha256": file_sha256(path),
        "examples": len(targets), "loss": loss,
        "accuracy": overall_acc, "macro_f1": macro_f1,
        "checkpoint_saved_validation_loss": float(checkpoint["validation_loss"]),
        "top_confusions": mistakes,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"N={summary['examples']} loss={loss:.4f} accuracy={overall_acc:.4f} macro_F1={macro_f1:.4f}")
    print("Weakest breeds by recall:")
    for item in sorted(per_class, key=lambda row: row["recall"])[:5]:
        print(f"  {item['breed']:<24} recall={item['recall']:.3f} support={item['support']}")
    print("Most frequent directed confusion pairs:")
    for item in mistakes[:5]:
        print(f"  {item['true']} -> {item['predicted']}: {item['count']}")
    print(f"Saved outputs in: {output_dir}")
    return summary


def validate(path: Path) -> None:
    summary = evaluate_checkpoint(
        path, split="validation", output_dir=validation_dir(path)
    )
    delta = abs(summary["loss"] - summary["checkpoint_saved_validation_loss"])
    if delta > 0.01:
        print("WARNING: recomputed validation loss differs from checkpoint metadata by "
              f"{delta:.4f}. Verify package versions and original evaluation pipeline before locking.")
    print("Official test set was NOT accessed.")


def lock_selection(path: Path) -> None:
    if SELECTION_PATH.exists():
        raise FileExistsError(f"Selection already locked: {SELECTION_PATH}")
    checkpoint = load_checkpoint(path)
    validation_path = validation_dir(path) / "summary.json"
    if not validation_path.exists():
        raise FileNotFoundError("Run `validate --checkpoint ...` for this candidate first")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    digest = file_sha256(path)
    if validation["checkpoint_sha256"] != digest:
        raise ValueError("Checkpoint changed since validation; rerun validation before locking")
    if abs(float(validation["loss"]) - float(checkpoint["validation_loss"])) > 0.01:
        raise ValueError("Recomputed validation loss differs from checkpoint metadata; investigate")
    SELECTION_PATH.parent.mkdir(parents=True, exist_ok=True)
    selection = {
        "selected_checkpoint": str(path.resolve()),
        "checkpoint_sha256": digest,
        "validation_loss": validation["loss"],
        "validation_accuracy": validation["accuracy"],
        "selection_rule": "Chosen using validation analysis only; no official test feedback",
    }
    SELECTION_PATH.write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(f"Selection locked at {SELECTION_PATH}")
    print("The next command should be the ONE final held-out test evaluation.")


def run_final_test() -> None:
    if not SELECTION_PATH.exists():
        raise FileNotFoundError("No selected_model.json. Lock a validated checkpoint first")
    if FINAL_DIR.exists():
        raise FileExistsError(f"Final test results already exist: {FINAL_DIR}. Do not repeat test-guided tuning")
    selection = json.loads(SELECTION_PATH.read_text(encoding="utf-8"))
    path = Path(selection["selected_checkpoint"])
    if file_sha256(path) != selection["checkpoint_sha256"]:
        raise ValueError("Selected checkpoint has changed since locking")
    # Only this mode opens the official test split.
    summary = evaluate_checkpoint(path, split="test", output_dir=FINAL_DIR)
    print("\nFINAL HELD-OUT TEST RESULT (do not use for model selection):")
    print(f"Accuracy={summary['accuracy']:.4f}, loss={summary['loss']:.4f}, macro F1={summary['macro_f1']:.4f}")

def compare_predictions(
    head_csv: str,
    fine_tuned_csv: str,
) -> None:
    """Compare predictions for the same validation examples, regardless of row order."""
    def read_predictions(filename: str) -> dict[int, dict]:
        rows = {}
        required = {"source_index", "true_id", "true_breed", "pred_id"}
        with Path(filename).open(newline="", encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if not required.issubset(reader.fieldnames or []):
                raise ValueError(f"{filename}: missing prediction columns {sorted(required)}")
            for line, row in enumerate(reader, start=2):
                try:
                    source_index = int(row["source_index"])
                    true_id, pred_id = int(row["true_id"]), int(row["pred_id"])
                except (TypeError, ValueError) as exc:
                    raise ValueError(f"{filename}:{line}: invalid integer ID") from exc
                if source_index < 0 or not (0 <= true_id < 37 and 0 <= pred_id < 37):
                    raise ValueError(f"{filename}:{line}: ID out of range")
                if source_index in rows:
                    raise ValueError(f"{filename}: duplicate source_index {source_index}")
                rows[source_index] = {
                    "true_id": true_id, "true_breed": row["true_breed"],
                    "correct": pred_id == true_id,
                }
        if not rows:
            raise ValueError(f"{filename}: no predictions")
        return rows

    head = read_predictions(head_csv)
    fine_tuned = read_predictions(fine_tuned_csv)
    if head.keys() != fine_tuned.keys():
        raise ValueError("Prediction files must contain exactly the same source indices")
    both_correct = both_wrong = fixed = regressed = 0
    for source_index, before in head.items():
        after = fine_tuned[source_index]
        if (before["true_id"], before["true_breed"]) != (after["true_id"], after["true_breed"]):
            raise ValueError(f"True label differs for source_index {source_index}")
        if before["correct"] and after["correct"]:
            both_correct += 1
        elif not before["correct"] and not after["correct"]:
            both_wrong += 1
        elif after["correct"]:
            fixed += 1
        else:
            regressed += 1
    n = len(head)
    print(f"Compared {n} matching examples")
    print(f"Head-only accuracy: {(both_correct + regressed) / n:.2%}")
    print(f"Fine-tuned accuracy: {(both_correct + fixed) / n:.2%}")
    print(f"Accuracy change: {100 * (fixed - regressed) / n:+.2f} percentage points")
    print(f"Both correct: {both_correct}; both wrong: {both_wrong}")
    print(f"Mistakes fixed by fine-tuning: {fixed}")
    print(f"New mistakes after fine-tuning: {regressed}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("validate", "lock", "final-test", "compare"))
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--head-csv", help="Head-only validation predictions.csv")
    parser.add_argument("--fine-tuned-csv", help="Fine-tuned validation predictions.csv")
    args = parser.parse_args()
    if args.action in ("validate", "lock") and args.checkpoint is None:
        parser.error("--checkpoint is required for validate and lock")
    if args.action == "compare" and (args.head_csv is None or args.fine_tuned_csv is None):
        parser.error("--head-csv and --fine-tuned-csv are required for compare")
    if args.action == "validate":
        validate(args.checkpoint)
    elif args.action == "lock":
        lock_selection(args.checkpoint)
    elif args.action == "compare":
        compare_predictions(args.head_csv, args.fine_tuned_csv)
    else:
        run_final_test()


if __name__ == "__main__":
    main()
