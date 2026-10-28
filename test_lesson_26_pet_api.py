"""Tests for Lesson 12; use a synthetic checkpoint, never real test images.

    python -m pytest -q test_lesson_26_pet_api.py
"""

from io import BytesIO
import json
from pathlib import Path

import pytest
import torch
from fastapi.testclient import TestClient
from PIL import Image
from torch import nn
from torchvision.models import mobilenet_v3_small

from lesson_26_pet_api import create_app, sha256_file, PREPROCESSING_ID


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    folder = tmp_path_factory.mktemp("pet-api")
    checkpoint_path = folder / "synthetic_checkpoint.pt"
    selection_path = folder / "selected_model.json"
    torch.manual_seed(123)
    model = mobilenet_v3_small(weights=None)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 37)
    class_names = [f"breed_{i:02d}" for i in range(37)]
    torch.save({
        "checkpoint_version": 1,
        "architecture": "mobilenet_v3_small",
        "num_classes": 37,
        "class_names": class_names,
        "preprocessing": PREPROCESSING_ID,
        "model_state_dict": model.state_dict(),
        "validation_loss": 0.3431,
        "validation_accuracy": 0.88,
    }, checkpoint_path)
    selection_path.write_text(json.dumps({
        "selected_checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "validation_loss": 0.3431,
        "validation_accuracy": 0.88,
    }), encoding="utf-8")
    with TestClient(create_app(selection_path, device=torch.device("cpu"))) as test_client:
        yield test_client


def image_bytes(format_name="PNG"):
    image = Image.new("RGB", (48, 48), (220, 110, 40))
    buffer = BytesIO()
    image.save(buffer, format=format_name)
    return buffer.getvalue()


def test_health_and_metadata(client):
    assert client.get("/health").json() == {"status": "ok", "model": "loaded"}
    info = client.get("/model-info").json()
    assert info["architecture"] == "mobilenet_v3_small"
    assert info["num_classes"] == 37
    assert len(info["checkpoint_sha256"]) == 64


def test_predict_returns_valid_probabilities(client):
    response = client.post(
        "/predict",
        files={"file": ("pet.png", image_bytes(), "image/png")},
    )
    assert response.status_code == 200, response.text
    prediction = response.json()
    assert prediction["predicted_breed"] == prediction["top_predictions"][0]["breed"]
    assert 0.0 <= prediction["confidence"] <= 1.0
    top = prediction["top_predictions"]
    assert len(top) == 3
    assert all(top[i]["probability"] >= top[i + 1]["probability"] for i in range(2))


def test_rejects_nonimage_and_bogus_image(client):
    response = client.post(
        "/predict", files={"file": ("notes.txt", b"not an image", "text/plain")}
    )
    assert response.status_code == 415
    response = client.post(
        "/predict", files={"file": ("bad.png", b"not a valid PNG", "image/png")}
    )
    assert response.status_code == 422


def test_rejects_oversized_file(client):
    response = client.post(
        "/predict",
        files={"file": ("huge.png", b"x" * (5 * 1024 * 1024 + 1), "image/png")},
    )
    assert response.status_code == 413


def test_missing_selection_fails_at_startup(tmp_path):
    with pytest.raises(FileNotFoundError, match="model-selection lock"):
        with TestClient(create_app(tmp_path / "absent.json", device=torch.device("cpu"))):
            pass
