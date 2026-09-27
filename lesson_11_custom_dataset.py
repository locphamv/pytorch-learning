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


class StudentDataset(Dataset):
    def __init__(
        self,
        csv_path: Path,
    ):
        self.data = pd.read_csv(
            csv_path
        )

        if self.data.empty:
            raise ValueError(
                "Dataset is empty"
            )

        required_columns = (
            set(FEATURE_NAMES)
            | {TARGET_NAME}
        )

        missing_columns = (
            required_columns
            - set(self.data.columns)
        )

        if missing_columns:
            raise ValueError(
                "Dataset is missing columns: "
                f"{sorted(missing_columns)}"
            )

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

        return features, target


class StudentClassifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        return self.network(x)


def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


dataset = StudentDataset(
    Path("data/students.csv")
)

loader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
)

device = get_device()

model = StudentClassifier().to(
    device
)

loss_function = nn.BCEWithLogitsLoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=0.01,
)

epochs = 200

model.train()

for epoch in range(epochs):
    for batch_X, batch_y in loader:
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


model.eval()

correct = 0
total = 0

with torch.inference_mode():
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)

        batch_y = (
            batch_y
            .unsqueeze(1)
            .to(device)
        )

        logits = model(batch_X)

        probabilities = torch.sigmoid(
            logits
        )

        predictions = (
            probabilities >= 0.5
        ).float()

        correct += (
            predictions == batch_y
        ).sum().item()

        total += batch_y.numel()


accuracy = correct / total

print(
    "Training accuracy:",
    round(accuracy, 4),
)

