import argparse
import csv
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import OxfordIIITPet
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.transforms import v2

from lesson_23_pet_capstone_baseline import evaluate, get_device, train_one_epoch

SEED = 42
BASELINE_PATH = Path("runs/pet_breeds_baseline/best-model.pt")
DATA_DIR = Path("data")
RUN_ROOT = Path("runs")

# Keep the comparison budgets explicit. Baseline was 3 epochs at batch size 16.

CONFIGS = {
    "augmentation": {
        "epochs": 3, "batch_size": 16, "head_lr": 1e-3,
        "tail_lr": None, "augment": True,
        "run_dir": "pet_breeds_augmented",
    },
    "head-continuation": {
        "epochs": 2, "batch_size": 8, "head_lr": 1e-4,
        "tail_lr": None, "augment": False,
        "run_dir": "pet_breeds_head_continuation",
    },
    "fine-tune": {
        "epochs": 2, "batch_size": 8, "head_lr": 1e-4,
        "tail_lr": 1e-5, "augment": False,
        "run_dir": "pet_breeds_tail_finetune",
    },
}


def load_baseline_metadata() -> dict:
    if not BASELINE_PATH.is_file():
        raise FileNotFoundError(
            f"Missing {BASELINE_PATH}. Run Lesson 23 first; "
            "do not substitute an untrained model."
        )
    checkpoint = torch.load(
        BASELINE_PATH, map_location="cpu", weights_only=True)
    required = {
        "checkpoint_version", "architecture", "class_names", "num_classes",
        "train_indices", "val_indices", "model_state_dict",
        "validation_loss", "validation_accuracy", "preprocessing",
    }
    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Baseline checkpoint missing keys: {sorted(missing)}")
    if checkpoint["architecture"] != "mobilenet_v3_small":
        raise ValueError("Baseline checkpoint uses a different architecture")
    if checkpoint["preprocessing"] != "MobileNet_V3_Small_Weights.DEFAULT.transforms()":
        raise ValueError("Baseline checkpoint uses unexpected preprocessing")
    if len(checkpoint["class_names"]) != checkpoint["num_classes"]:
        raise ValueError(
            "Invalid number of class names in baseline checkpoint")
    return checkpoint


def make_loaders(
    checkpoint: dict, augment: bool, batch_size: int
) -> tuple[DataLoader, DataLoader]:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    evaluation_transform = weights.transforms()

    if augment:
        training_transform = v2.Compose([
            v2.ToImage(),
            v2.RandomResizedCrop(
                size=(224, 224), scale=(0.80, 1.0),
                ratio=(0.90, 1.10), antialias=True,
            ),
            v2.RandomHorizontalFlip(p=0.5),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(
                mean=evaluation_transform.mean,
                std=evaluation_transform.std,
            ),
        ])
    else:
        training_transform = evaluation_transform

    train_source = OxfordIIITPet(
        root=DATA_DIR, split="trainval", target_types="category",
        download=False, transform=training_transform,
    )

    validation_source = OxfordIIITPet(
        root=DATA_DIR, split="trainval", target_types="category",
        download=False, transform=evaluation_transform,
    )

    train_idx = checkpoint["train_indices"]
    val_idx = checkpoint["val_indices"]
    n = len(train_source)
    assert len(validation_source) == n
    if (len(set(train_idx)) != len(train_idx) or
        len(set(val_idx)) != len(val_idx) or
        set(train_idx) & set(val_idx) or
            set(train_idx) | set(val_idx) != set(range(n))):
        raise ValueError(
            "Invalid, overlapping, or incomplete saved data split"
        )
    if list(train_source.classes) != list(checkpoint["class_names"]):
        raise ValueError("Dataset class order differs from baseline")

    train_loader = DataLoader(
        Subset(train_source, train_idx), batch_size=batch_size,
        shuffle=True, num_workers=0,
        generator=torch.Generator().manual_seed(SEED+1),
    )

    val_loader = DataLoader(
        Subset(validation_source, val_idx), batch_size=batch_size,
        shuffle=False, num_workers=0,
        generator=torch.Generator().manual_seed(SEED + 2),
    )

    return train_loader, val_loader


def make_model(checkpoint: dict, experiment: str):
    num_classes = checkpoint["num_classes"]
    if experiment == "augmentation":
        model = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
    else:
        model = mobilenet_v3_small(weights=None)
        input_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(input_features, num_classes)
        model.load_state_dict(checkpoint["model_state_dict"])

    for parameter in model.parameters():
        parameter.requires_grad_(False)
    for parameter in model.classifier[-1].parameters():
        parameter.requires_grad_(True)
    if experiment == "fine-tune":
        for parameter in model.features[-1].parameters():
            parameter.requires_grad_(True)
    return model


def make_optimizer(model: nn.Module, config: dict):
    groups = [
        {
            "params": model.classifier[-1].parameters(),
            "lr": config["head_lr"],
        }
    ]

    if config["tail_lr"] is not None:
        groups.append(
            {
                "params": model.features[-1].parameters(),
                "lr": config["tail_lr"],
            }
        )

    return torch.optim.Adam(groups)


def save_best(
    model: nn.Module, source: dict, experiment: str,
    config: dict, epoch: int, val_loss: float,
    val_accuracy: float, path: Path,
):
    cpu_state = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }
    torch.save({
        "checkpoint_version": 1,
        "architecture": "mobilenet_v3_small",
        "class_names": source["class_names"],
        "num_classes": source["num_classes"],
        "preprocessing": source["preprocessing"],
        "train_indices": source["train_indices"],
        "val_indices": source["val_indices"],
        "split_seed": SEED,
        "experiment": experiment,
        "augmentation": config["augment"],
        "head_lr": config["head_lr"],
        "tail_lr": config["tail_lr"],
        "epoch": epoch,
        "validation_loss": val_loss,
        "validation_accuracy": val_accuracy,
        "model_state_dict": cpu_state,
    }, path)

def run(experiment: str):
    torch.manual_seed(SEED)
    cfg = CONFIGS[experiment]
    source = load_baseline_metadata()
    train_loader, val_loader = make_loaders(
        source, augment=cfg["augment"], batch_size=cfg["batch_size"]
    )
    device = get_device()
    model = make_model(source, experiment).to(device)
    loss_function = nn.CrossEntropyLoss()
    optimizer = make_optimizer(model, cfg)
    run_dir = RUN_ROOT / cfg["run_dir"]
    run_dir.mkdir(parents=True, exist_ok = True)
    output_path = run_dir / "best_model.pt"

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # Avoid consuming training shuffle or random augmentation during inspection.
    sample_images, sample_labels = next(iter(val_loader))
    assert sample_images.shape[1:] == (3, 224, 224)
    assert sample_labels.ndim == 1
    print(f"Experiment: {experiment} | Device: {device}")
    print(f"Train: {len(train_loader.dataset)} | Validation: {len(val_loader.dataset)}")
    print(f"Trainable parameters: {trainable}")
    print("Official test split: untouched")
    best_loss = math.inf
    best_accuracy = 0.0
    best_epoch = -1

    if experiment != "augmentation":
        # Evaluate the shared starting checkpoint; it is the epoch-zero candidate.
        initial_loss, initial_acc = evaluate(
            model, val_loader, loss_function, device
        )
        # Catch accidental preprocessing / dataset mismatch before comparing.
        if abs(initial_loss - float(source["validation_loss"])) > 0.02:
            raise RuntimeError(
                "Baseline validation loss disagrees with saved result; "
                "check dataset, preprocessing, and checkpoint."
            )
        best_loss, best_accuracy, best_epoch = initial_loss, initial_acc, 0
        save_best(model, source, experiment, cfg, 0,
                  initial_loss, initial_acc, output_path)
        print(f"Epoch 00 (same baseline): val_loss={initial_loss:.4f} "
              f"val_accuracy={initial_acc:.4f}")

    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        # Imported Lesson 23 training function uses model.eval() intentionally:
        # BatchNorm buffers + dropout stay fixed while autograd trains parameters.
        train_loss = train_one_epoch(
            model, train_loader, loss_function, optimizer, device
        )
        val_loss, val_accuracy = evaluate(
            model, val_loader, loss_function, device
        )
        improved = val_loss < best_loss
        if improved:
            best_loss, best_accuracy, best_epoch = val_loss, val_accuracy, epoch
            save_best(model, source, experiment, cfg, epoch,
                      val_loss, val_accuracy, output_path)
        history.append({
            "epoch": epoch, "train_loss": train_loss,
            "val_loss": val_loss, "val_accuracy": val_accuracy,
            "new_best": int(improved),
        })
        print(f"Epoch {epoch:02d}: train_loss={train_loss:.4f} "
              f"val_loss={val_loss:.4f} val_accuracy={val_accuracy:.4f} "
              f"new_best={improved}")

    with (run_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    summary = {
        "experiment": experiment,
        "best_epoch": best_epoch,
        "best_val_loss": best_loss,
        "best_val_accuracy": best_accuracy,
        "baseline_val_loss": float(source["validation_loss"]),
        "baseline_val_accuracy": float(source["validation_accuracy"]),
        "trainable_parameters": trainable,
        "epochs": cfg["epochs"],
        "batch_size": cfg["batch_size"],
        "head_lr": cfg["head_lr"],
        "tail_lr": cfg["tail_lr"],
        "augmentation": cfg["augment"],
        "test_evaluated": False,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    # Verify the saved state can be loaded into a fresh architecture.
    saved = torch.load(output_path, map_location="cpu", weights_only=True)
    check_model = mobilenet_v3_small(weights=None)
    check_model.classifier[-1] = nn.Linear(
        check_model.classifier[-1].in_features, source["num_classes"]
    )
    check_model.load_state_dict(saved["model_state_dict"])
    print(f"\nBest validation epoch: {best_epoch}")
    print(f"Best validation loss: {best_loss:.4f}")
    print(f"Best validation accuracy: {best_accuracy:.4f}")
    print(f"Checkpoint reload: OK | Saved to: {output_path}")
    print("Official test split: NOT evaluated")



def compare_experiments(
    first_summary: Path,
    second_summary: Path,
) -> None:
    if not first_summary.is_file():
        raise FileNotFoundError(f"Missing summary file: {first_summary}")
    if not second_summary.is_file():
        raise FileNotFoundError(f"Missing summary file: {second_summary}")

    with first_summary.open("r", encoding="utf-8") as f:
        first = json.load(f)
    with second_summary.open("r", encoding="utf-8") as f:
        second = json.load(f)

    exp1 = first.get("experiment", first_summary.parent.name)
    exp2 = second.get("experiment", second_summary.parent.name)

    acc1, acc2 = first["best_val_accuracy"], second["best_val_accuracy"]
    loss1, loss2 = first["best_val_loss"], second["best_val_loss"]
    delta_acc = acc2 - acc1
    delta_loss = loss2 - loss1

    print("=" * 65)
    print(f"EXPERIMENT COMPARISON: {exp1} vs {exp2}")
    print("=" * 65)
    print(f"{'Metric':<25} | {exp1:<16} | {exp2:<16}")
    print("-" * 65)
    print(f"{'Best Epoch':<25} | {first['best_epoch']:<16} | {second['best_epoch']:<16}")
    print(f"{'Best Val Accuracy':<25} | {acc1*100:>6.2f}%          | {acc2*100:>6.2f}%")
    print(f"{'Best Val Loss':<25} | {loss1:<16.4f} | {loss2:<16.4f}")
    print(f"{'Head LR':<25} | {str(first.get('head_lr')):<16} | {str(second.get('head_lr')):<16}")
    print(f"{'Tail LR':<25} | {str(first.get('tail_lr')):<16} | {str(second.get('tail_lr')):<16}")
    print(f"{'Augmentation':<25} | {str(first.get('augmentation')):<16} | {str(second.get('augmentation')):<16}")
    print("-" * 65)
    print(f"Delta Accuracy ({exp2} - {exp1}): {delta_acc * 100:+.2f}%")
    print(f"Delta Loss     ({exp2} - {exp1}): {delta_loss:+.4f}")

    if delta_acc > 0:
        print(f"Outcome: {exp2} improved accuracy by {delta_acc * 100:.2f} percentage points.")
    elif delta_acc < 0:
        print(f"Outcome: {exp1} had higher accuracy by {(-delta_acc) * 100:.2f} percentage points.")
    else:
        print("Outcome: Both experiments achieved identical validation accuracy.")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=tuple(CONFIGS))
    parser.add_argument(
        "--compare",
        nargs=2,
        metavar=("FIRST_SUMMARY", "SECOND_SUMMARY"),
        help="Compare two summary.json files",
    )
    args = parser.parse_args()

    if args.compare:
        compare_experiments(Path(args.compare[0]), Path(args.compare[1]))
    elif args.experiment:
        run(args.experiment)
    else:
        parser.error("Specify either --experiment or --compare")


if __name__ == "__main__":
    main()
