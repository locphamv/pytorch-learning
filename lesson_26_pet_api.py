"""CV Lesson 12: serve the locked Oxford-IIIT Pet classifier locally.

From the pytorch-learning project root, after Lesson 11's `lock` step:
    python -m uvicorn lesson_26_pet_api:app --host 127.0.0.1 --port 8000

Requirements:
    python -m pip install fastapi 'uvicorn[standard]' python-multipart pillow

A missing or modified checkpoint makes application startup fail closed.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from dataclasses import dataclass
from io import BytesIO
import hashlib
import json
from pathlib import Path
from threading import Lock

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image, ImageOps, UnidentifiedImageError
from pydantic import BaseModel
from torch import nn
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small


SELECTION_PATH = Path("runs/pet_breeds_evaluation/selected_model.json")
PREPROCESSING_ID = "MobileNet_V3_Small_Weights.DEFAULT.transforms()"
MAX_UPLOAD_BYTES = 5 * 1024 * 1024  # 5 MiB
MAX_IMAGE_PIXELS = 12_000_000
ALLOWED_MEDIA_TYPES = {"image/jpeg", "image/png", "image/webp"}
ALLOWED_IMAGE_FORMATS = ("JPEG", "PNG", "WEBP")


class BreedProbability(BaseModel):
    breed: str
    probability: float


class PredictionResponse(BaseModel):
    predicted_breed: str
    confidence: float
    top_predictions: list[BreedProbability]


class ModelInfo(BaseModel):
    architecture: str
    num_classes: int
    preprocessing: str
    checkpoint_sha256: str
    validation_loss: float
    validation_accuracy: float
    device: str


@dataclass
class Predictor:
    model: nn.Module
    device: torch.device
    class_names: list[str]
    sha256: str
    validation_loss: float
    validation_accuracy: float
    lock: Lock

    def predict(self, image: Image.Image) -> PredictionResponse:
        transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
        tensor = transform(image).unsqueeze(0).to(self.device)
        # This lock limits concurrent inference on memory-constrained hardware.
        with self.lock, torch.inference_mode():
            logits = self.model(tensor)  # [1, 37]
            probabilities = torch.softmax(logits, dim=1)[0].cpu()
        values, indices = probabilities.topk(3)
        top = [
            BreedProbability(
                breed=self.class_names[int(index)],
                probability=float(value),
            )
            for value, index in zip(values.tolist(), indices.tolist())
        ]
        return PredictionResponse(
            predicted_breed=top[0].breed,
            confidence=top[0].probability,
            top_predictions=top,
        )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def select_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def load_predictor(
    selection_path: Path = SELECTION_PATH,
    device: torch.device | None = None,
) -> Predictor:
    if not selection_path.is_file():
        raise FileNotFoundError(
            f"Missing model-selection lock: {selection_path}. "
            "Complete Lesson 11 validation and lock the chosen checkpoint first."
        )
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    checkpoint_path = Path(selection["selected_checkpoint"])
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Missing selected checkpoint: {checkpoint_path}")

    current_sha256 = sha256_file(checkpoint_path)
    if current_sha256 != selection["checkpoint_sha256"]:
        raise ValueError("Checkpoint SHA-256 differs from the locked selection")

    # Our checkpoints contain tensors and primitive metadata, not Python objects.
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint is not a dictionary")
    if checkpoint.get("checkpoint_version") != 1:
        raise ValueError("Unsupported checkpoint version")
    if checkpoint.get("architecture") != "mobilenet_v3_small":
        raise ValueError("Unsupported model architecture")
    if checkpoint.get("preprocessing") != PREPROCESSING_ID:
        raise ValueError("Checkpoint preprocessing does not match inference")
    class_names = checkpoint.get("class_names")
    if (
        not isinstance(class_names, list)
        or len(class_names) != 37
        or not all(isinstance(name, str) and name for name in class_names)
        or len(set(class_names)) != len(class_names)
        or checkpoint.get("num_classes") != len(class_names)
    ):
        raise ValueError("Invalid 37-class checkpoint metadata")
    if abs(float(checkpoint["validation_loss"]) - float(selection["validation_loss"])) > 0.01:
        raise ValueError("Selection metadata disagrees with checkpoint validation loss")

    model = mobilenet_v3_small(weights=None)  # no network download at startup
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(class_names))
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    runtime_device = select_device() if device is None else device
    model = model.to(runtime_device).eval()
    return Predictor(
        model=model,
        device=runtime_device,
        class_names=class_names,
        sha256=current_sha256,
        validation_loss=float(checkpoint["validation_loss"]),
        validation_accuracy=float(checkpoint["validation_accuracy"]),
        lock=Lock(),
    )


def decode_uploaded_image(raw: bytes) -> Image.Image:
    try:
        with Image.open(BytesIO(raw), formats=ALLOWED_IMAGE_FORMATS) as image:
            if image.width * image.height > MAX_IMAGE_PIXELS:
                raise HTTPException(status_code=413, detail="Image has too many pixels")
            # Decode data and apply the camera's EXIF orientation before converting.
            image.load()
            return ImageOps.exif_transpose(image).convert("RGB")
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise HTTPException(status_code=422, detail="Invalid or damaged image") from exc
    except Image.DecompressionBombError as exc:
        raise HTTPException(status_code=413, detail="Image has too many pixels") from exc


def create_app(
    selection_path: Path = SELECTION_PATH,
    device: torch.device | None = None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = load_predictor(selection_path, device)
        yield
        del app.state.predictor

    app = FastAPI(title="Pet Breed Classifier", version="1.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": "loaded"}

    @app.get("/model-info", response_model=ModelInfo)
    def model_info() -> ModelInfo:
        predictor: Predictor = app.state.predictor
        return ModelInfo(
            architecture="mobilenet_v3_small",
            num_classes=len(predictor.class_names),
            preprocessing=PREPROCESSING_ID,
            checkpoint_sha256=predictor.sha256,
            validation_loss=predictor.validation_loss,
            validation_accuracy=predictor.validation_accuracy,
            device=str(predictor.device),
        )

    @app.post("/predict", response_model=PredictionResponse)
    def predict(file: UploadFile = File(...)) -> PredictionResponse:
        try:
            if file.content_type not in ALLOWED_MEDIA_TYPES:
                raise HTTPException(status_code=415, detail="Upload JPEG, PNG, or WebP")
            # Never call read() without a bound: avoid unlimited buffering.
            raw = file.file.read(MAX_UPLOAD_BYTES + 1)
            if len(raw) > MAX_UPLOAD_BYTES:
                raise HTTPException(status_code=413, detail="File exceeds 5 MiB")
            if not raw:
                raise HTTPException(status_code=422, detail="Empty image")
            image = decode_uploaded_image(raw)
            return app.state.predictor.predict(image)
        finally:
            file.file.close()

    return app


app = create_app()
