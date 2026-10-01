from collections.abc import Callable
from pathlib import Path

import pandas as pd
import torch
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


class ClampFeatures:
    def __init__(
        self,
        minimum: float,
        maximum: float,
    ):
        self.minimum = minimum
        self.maximum = maximum

    def __call__(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        return torch.clamp(
            features,
            min=self.minimum,
            max=self.maximum,
        )


class Compose:
    def __init__(
        self,
        transforms: list[
            Callable[
                [torch.Tensor],
                torch.Tensor,
            ]
        ],
    ):
        self.transforms = transforms

    def __call__(
        self,
        features: torch.Tensor,
    ) -> torch.Tensor:
        for transform in self.transforms:
            features = transform(
                features
            )

        return features


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


def load_student_data(
    csv_path: Path,
) -> pd.DataFrame:
    data = pd.read_csv(
        csv_path
    )

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
        FEATURE_NAMES
        + [TARGET_NAME]
    ].isnull().any().any():
        raise ValueError(
            "Dataset contains missing values"
        )

    return data


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


data = load_student_data(
    Path("data/students.csv")
)

generator = torch.Generator()
generator.manual_seed(42)

indices = torch.randperm(
    len(data),
    generator=generator,
)

train_indices = indices[:12]
validation_indices = indices[12:14]
test_indices = indices[14:]

train_data = data.iloc[
    train_indices.tolist()
].copy()

validation_data = data.iloc[
    validation_indices.tolist()
].copy()

test_data = data.iloc[
    test_indices.tolist()
].copy()


feature_mean, feature_std = (
    calculate_feature_statistics(
        train_data
    )
)

feature_transform = Compose(
    [
        StandardizeFeatures(
            feature_mean,
            feature_std,
        ),
        ClampFeatures(
            minimum=-3.0,
            maximum=3.0,
        ),
    ]
)

train_dataset = StudentDataset(
    train_data,
    transform=feature_transform,
)

validation_dataset = StudentDataset(
    validation_data,
    transform=feature_transform,
)

test_dataset = StudentDataset(
    test_data,
    transform=feature_transform,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=4,
    shuffle=True,
)


transformed_training_features = (
    torch.stack(
        [
            train_dataset[index][0]
            for index in range(
                len(train_dataset)
            )
        ]
    )
)

batch_X, batch_y = next(
    iter(train_loader)
)

print(
    "Transformed min:",
    transformed_training_features.min().item(),
)

print(
    "Transformed max:",
    transformed_training_features.max().item(),
)

print(
    "Batch X shape:",
    batch_X.shape,
)

print(
    "Batch y shape:",
    batch_y.shape,
)
