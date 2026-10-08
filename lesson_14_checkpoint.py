from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import torch
from torch import nn
from torch.utils.data import (
    DataLoader,
    Dataset,
)


FEATURE_NAMES = [
    "study_hours",
    "absences",
    "previous_score",
    "sleep_hours",
]

TARGET_NAME = "passed"

BATCH_SIZE = 4
EPOCHS = 200
LEARNING_RATE = 0.01

HIDDEN_FEATURES = 8
PREDICTION_THRESHOLD = 0.5
CHECKPOINT_VERSION = 1


class StandardizeFeatures:
    def __init__(
        self,
        mean: torch.Tensor,
        std: torch.Tensor,
    ):
        if torch.any(std == 0):
            raise ValueError(
                "Feature standard deviation "
                "cannot be zero"
            )

        self.mean = mean
        self.std = std

    def __call__(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return (
            features - self.mean
        ) / self.std


class StudentDataset(Dataset):
    def __init__(
        self,
        data: pd.DataFrame,
        transform: Callable[
            [torch.Tensor],
            torch.Tensor,
        ] | None = None,
    ):
        self.data = data.reset_index(
            drop=True
        )
        self.transform = transform

    def __len__(
        self,
    ) -> int:
        return len(self.data)

    def __getitem__(
        self,
        index: int,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
    ]:
        row = self.data.iloc[index]

        features = torch.tensor(
            [
                row[name]
                for name in FEATURE_NAMES
            ],
            dtype=torch.float32,
        )

        target = torch.tensor(
            row[TARGET_NAME],
            dtype=torch.float32,
        )

        if self.transform is not None:
            features = self.transform(
                features
            )

        return features, target


class StudentClassifier(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_features: int,
    ):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(
                input_features,
                hidden_features,
            ),
            nn.ReLU(),
            nn.Linear(
                hidden_features,
                1,
            ),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


@dataclass
class LoadedModel:
    model: StudentClassifier
    transform: StandardizeFeatures
    feature_names: list[str]
    prediction_threshold: float


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


def load_student_data(
    csv_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(csv_path)

    if data.empty:
        raise ValueError(
            "Dataset is empty"
        )

    required_columns = (
        set(FEATURE_NAMES)
        | {TARGET_NAME}
    )

    missing_columns = (
        required_columns
        - set(data.columns)
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    if data[
        FEATURE_NAMES + [TARGET_NAME]
    ].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    target_values = set(
        data[TARGET_NAME]
        .unique()
        .tolist()
    )

    if target_values != {0, 1}:
        raise ValueError(
            "Target must contain "
            "exactly classes 0 and 1"
        )

    return data


def split_data(
    data: pd.DataFrame,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    generator = torch.Generator()
    generator.manual_seed(42)

    indices = torch.randperm(
        len(data),
        generator=generator,
    )

    train_end = int(
        len(data) * 0.75
    )

    validation_end = int(
        len(data) * 0.875
    )

    train_indices = indices[:train_end]

    validation_indices = indices[
        train_end:validation_end
    ]

    test_indices = indices[
        validation_end:
    ]

    train_data = data.iloc[
        train_indices.tolist()
    ].copy()

    validation_data = data.iloc[
        validation_indices.tolist()
    ].copy()

    test_data = data.iloc[
        test_indices.tolist()
    ].copy()

    return (
        train_data,
        validation_data,
        test_data,
    )


def calculate_feature_statistics(
    data: pd.DataFrame,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
]:
    features = torch.tensor(
        data[
            FEATURE_NAMES
        ].to_numpy(),
        dtype=torch.float32,
    )

    return (
        features.mean(dim=0),
        features.std(dim=0),
    )


def save_checkpoint(
    model: nn.Module,
    checkpoint_path: Path,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
) -> None:
    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "checkpoint_version": (
            CHECKPOINT_VERSION
        ),
        "model_state_dict": (
            model.state_dict()
        ),
        "model_config": {
            "input_features": (
                len(FEATURE_NAMES)
            ),
            "hidden_features": (
                HIDDEN_FEATURES
            ),
        },
        "feature_names": FEATURE_NAMES,
        "feature_mean": (
            feature_mean.cpu()
        ),
        "feature_std": (
            feature_std.cpu()
        ),
        "prediction_threshold": (
            PREDICTION_THRESHOLD
        ),
    }

    torch.save(
        checkpoint,
        checkpoint_path,
    )


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> LoadedModel:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    if not isinstance(
        checkpoint,
        dict,
    ):
        raise ValueError(
            "Checkpoint must be a dictionary"
        )

    required_keys = {
        "checkpoint_version",
        "model_state_dict",
        "model_config",
        "feature_names",
        "feature_mean",
        "feature_std",
        "prediction_threshold",
    }

    missing_keys = (
        required_keys
        - checkpoint.keys()
    )

    if missing_keys:
        raise ValueError(
            "Checkpoint is missing keys: "
            f"{sorted(missing_keys)}"
        )

    if (
        checkpoint[
            "checkpoint_version"
        ]
        != CHECKPOINT_VERSION
    ):
        raise ValueError(
            "Unsupported checkpoint version"
        )

    model_config = checkpoint[
        "model_config"
    ]

    model = StudentClassifier(
        input_features=(
            model_config[
                "input_features"
            ]
        ),
        hidden_features=(
            model_config[
                "hidden_features"
            ]
        ),
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model = model.to(device)
    model.eval()

    transform = StandardizeFeatures(
        mean=checkpoint[
            "feature_mean"
        ],
        std=checkpoint[
            "feature_std"
        ],
    )

    return LoadedModel(
        model=model,
        transform=transform,
        feature_names=checkpoint[
            "feature_names"
        ],
        prediction_threshold=float(
            checkpoint[
                "prediction_threshold"
            ]
        ),
    )


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch_X, batch_y in dataloader:
        batch_X = batch_X.to(device)

        batch_y = (
            batch_y
            .unsqueeze(1)
            .to(device)
        )

        logits = model(batch_X)

        loss = loss_function(
            logits,
            batch_y,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_size = batch_X.size(0)

        total_loss += (
            loss.item()
            * batch_size
        )

        total_samples += batch_size

    return (
        total_loss
        / total_samples
    )


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    loss_function: nn.Module,
    device: torch.device,
) -> tuple[
    float,
    float,
]:
    model.eval()

    total_loss = 0.0
    total_samples = 0
    correct_predictions = 0

    with torch.inference_mode():
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)

            batch_y = (
                batch_y
                .unsqueeze(1)
                .to(device)
            )

            logits = model(batch_X)

            loss = loss_function(
                logits,
                batch_y,
            )

            probabilities = torch.sigmoid(
                logits
            )

            predictions = (
                probabilities
                >= PREDICTION_THRESHOLD
            ).float()

            batch_size = batch_X.size(0)

            total_loss += (
                loss.item()
                * batch_size
            )

            total_samples += batch_size

            correct_predictions += (
                predictions
                == batch_y
            ).sum().item()

    return (
        total_loss / total_samples,
        correct_predictions / total_samples,
    )


def predict_student(
    loaded_model: LoadedModel,
    feature_values: dict[
        str,
        float,
    ],
    device: torch.device,
) -> tuple[
    int,
    float,
]:
    raw_features = torch.tensor(
        [
            feature_values[name]
            for name
            in loaded_model.feature_names
        ],
        dtype=torch.float32,
    )

    features = (
        loaded_model
        .transform(
            raw_features
        )
    )

    features = (
        features
        .unsqueeze(0)
        .to(device)
    )

    with torch.inference_mode():
        logits = loaded_model.model(
            features
        )

        probability = torch.sigmoid(
            logits
        ).item()

    prediction = int(
        probability
        >= loaded_model
        .prediction_threshold
    )

    return (
        prediction,
        probability,
    )


data = load_student_data(
    Path("data/students.csv")
)

(
    train_data,
    validation_data,
    test_data,
) = split_data(data)

feature_mean, feature_std = (
    calculate_feature_statistics(
        train_data
    )
)

transform = StandardizeFeatures(
    feature_mean,
    feature_std,
)

train_dataset = StudentDataset(
    train_data,
    transform,
)

validation_dataset = StudentDataset(
    validation_data,
    transform,
)

test_dataset = StudentDataset(
    test_data,
    transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

device = get_device()

model = StudentClassifier(
    input_features=len(FEATURE_NAMES),
    hidden_features=HIDDEN_FEATURES,
).to(device)

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE,
)

checkpoint_path = Path(
    "models/"
    "student_classifier_checkpoint.pt"
)

best_validation_loss = float("inf")

for epoch in range(EPOCHS):
    train_loss = train_one_epoch(
        model,
        train_loader,
        loss_function,
        optimizer,
        device,
    )

    (
        validation_loss,
        validation_accuracy,
    ) = evaluate(
        model,
        validation_loader,
        loss_function,
        device,
    )

    if (
        validation_loss
        < best_validation_loss
    ):
        best_validation_loss = (
            validation_loss
        )

        save_checkpoint(
            model=model,
            checkpoint_path=checkpoint_path,
            feature_mean=feature_mean,
            feature_std=feature_std,
        )

    if (epoch + 1) % 20 == 0:
        print(
            f"Epoch {epoch + 1:03d} "
            f"train_loss={train_loss:.4f} "
            f"val_loss={validation_loss:.4f} "
            f"val_acc={validation_accuracy:.4f}"
        )


loaded_model = load_checkpoint(
    checkpoint_path,
    device,
)

(
    test_loss,
    test_accuracy,
) = evaluate(
    loaded_model.model,
    test_loader,
    loss_function,
    device,
)

print(
    "\nFinal test loss:",
    round(test_loss, 4),
)

print(
    "Final test accuracy:",
    round(test_accuracy, 4),
)


prediction, probability = (
    predict_student(
        loaded_model=loaded_model,
        feature_values={
            "study_hours": 7.0,
            "absences": 1.0,
            "previous_score": 8.0,
            "sleep_hours": 7.5,
        },
        device=device,
    )
)

print(
    "Prediction:",
    prediction,
)

print(
    "Pass probability:",
    round(
        probability,
        4,
    ),
)
