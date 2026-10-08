# PyTorch Learning

A hands-on repository for learning **PyTorch and deep learning fundamentals** step by step.

This repo documents my progression from tensor basics to building a small end-to-end PyTorch training and inference pipeline with:

- custom `Dataset` and `DataLoader`
- model training and evaluation
- Apple MPS / CPU device support
- binary and multiclass classification
- model persistence with `state_dict`
- preprocessing transforms
- train / validation / test discipline
- best-checkpoint selection
- checkpoint packaging with preprocessing state
- reproducible raw-feature inference

The next phase of the roadmap is **Computer Vision**.

---

## Learning Roadmap

```text
PyTorch fundamentals
↓
Tensors, shapes, autograd
↓
nn.Module, losses, optimizers
↓
Neural networks
↓
Dataset / DataLoader
↓
Train / validation / test
↓
Device-aware training
↓
Model save / load
↓
Binary + multiclass classification
↓
Custom datasets
↓
Transforms + preprocessing
↓
Complete training pipeline
↓
Model + preprocessing checkpoint
↓
Computer Vision
```

---

## What I Have Covered

### 1. Tensors

- tensor creation
- `shape`, `ndim`, `numel`
- tensor dtypes
- indexing and reshaping
- reductions
- elementwise operations
- matrix multiplication
- CPU / MPS devices
- NumPy ↔ PyTorch conversion

### 2. Shapes and Broadcasting

- single samples vs batches
- matrix multiplication shape rules
- broadcasting
- multiple outputs
- `reshape`, `unsqueeze`, `squeeze`
- transpose
- debugging tensor shapes

### 3. Autograd and Gradient Descent

- `requires_grad=True`
- computation graphs
- `loss.backward()`
- parameter gradients
- manual gradient descent
- clearing accumulated gradients

### 4. `nn.Module`, Losses, and Optimizers

- custom PyTorch modules
- `nn.Linear`
- `forward()`
- `nn.MSELoss`
- `torch.optim.SGD`
- `model.train()`
- `model.eval()`
- `torch.inference_mode()`

### 5. Multi-Layer Neural Networks

- hidden layers
- ReLU
- nonlinear models
- XOR
- `nn.Sequential`
- binary logits
- `BCEWithLogitsLoss`

### 6. Dataset and DataLoader

- `TensorDataset`
- batching
- shuffling
- uneven final batches
- sample-weighted average loss

### 7. Train / Validation / Test

- training loop
- validation loop
- test evaluation
- generalization
- preventing test-set leakage

### 8. Device Management

- CPU fallback
- Apple MPS support
- moving models and batches to the same device
- portable inference
- moving tensors back to CPU for NumPy

### 9. Model Persistence

- `model.state_dict()`
- `torch.save`
- `torch.load`
- `load_state_dict`
- rebuilding a fresh model
- verifying restored predictions

### 10. Multiclass Classification

- logits shaped `[batch_size, num_classes]`
- `CrossEntropyLoss`
- class IDs as `torch.long`
- `argmax(dim=1)`
- softmax probabilities
- Adam optimizer

### 11. Custom Dataset from CSV

- subclassing `Dataset`
- `__len__`
- `__getitem__`
- reading tabular data with pandas
- CSV validation
- feature contracts
- binary classification from external data

### 12. Transforms and Preprocessing

- callable transform objects
- feature standardization
- training-only preprocessing statistics
- avoiding preprocessing leakage
- transform composition
- reusable preprocessing pipelines

### 13. Complete Training Pipeline

- data validation
- deterministic data splitting
- training statistics
- reusable train / evaluation functions
- validation loss
- best-model checkpoint selection
- final test evaluation

### 14. Model + Preprocessing Checkpoint

The final general PyTorch lesson packages the full inference contract into one checkpoint:

```text
architecture configuration
+
learned model weights
+
feature order
+
training mean
+
training standard deviation
+
prediction threshold
```

This allows a fresh Python process to restore the model and make predictions from **raw feature values** without manually recreating preprocessing.

---

## Example Pipeline

```text
students.csv
↓
validate data
↓
split train / validation / test
↓
calculate preprocessing from TRAIN only
↓
StudentDataset
↓
DataLoader
↓
StudentClassifier
↓
train
↓
validate
↓
save best checkpoint
↓
reload checkpoint
↓
final test evaluation
↓
raw-feature inference
```

---

## Current Student Features

The latest version uses:

```text
study_hours
absences
previous_score
sleep_hours
```

Target:

```text
passed
```

where:

```text
0 = fail
1 = pass
```

---

## Project Structure

```text
pytorch-learning/
├── data/
│   └── students.csv
├── models/
│   └── generated model checkpoints
├── lesson_01_*.py
├── lesson_02_*.py
├── ...
├── lesson_09_save_load.py
├── lesson_10_multiclass.py
├── lesson_11_custom_dataset.py
├── lesson_12_transforms.py
├── lesson_13_training_pipeline.py
├── lesson_14_checkpoint.py
├── .gitignore
└── README.md
```

Generated PyTorch model files are ignored by Git:

```text
models/*.pt
models/*.pth
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/locphamv/student-performance-prediction.git
cd pytorch-learning
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate it.

macOS / Linux:

```bash
source .venv/bin/activate
```

Windows:

```bash
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
python -m pip install torch numpy pandas
```

---

## Running a Lesson

For example:

```bash
python lesson_14_checkpoint.py
```

Earlier lessons can be run independently:

```bash
python lesson_09_save_load.py
python lesson_10_multiclass.py
python lesson_11_custom_dataset.py
python lesson_12_transforms.py
python lesson_13_training_pipeline.py
```

---

## Device Support

```python
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")
```

This lets the lessons use Apple MPS when available while falling back to CPU.

---

## Important Engineering Rules Practiced

- training, validation, and test sets have different jobs
- the test set should not be used for model selection
- preprocessing statistics must come from training data only
- model and input tensors must be on the same device
- raw logits should be passed directly to `BCEWithLogitsLoss` or `CrossEntropyLoss`
- generated model artifacts should generally not be committed to Git
- feature order is part of the inference contract
- a deployable model is more than just neural-network weights

---

## Checkpoint Design

The latest checkpoint stores a dictionary similar to:

```python
{
    "checkpoint_version": 1,
    "model_state_dict": ...,
    "model_config": {
        "input_features": 4,
        "hidden_features": 8,
    },
    "feature_names": [
        "study_hours",
        "absences",
        "previous_score",
        "sleep_hours",
    ],
    "feature_mean": ...,
    "feature_std": ...,
    "prediction_threshold": 0.5,
}
```

Loading reconstructs:

```text
checkpoint
↓
model architecture
↓
model weights
↓
preprocessing transform
↓
feature order
↓
prediction threshold
↓
ready-to-use inference pipeline
```

---

## Next Phase: Computer Vision

General PyTorch fundamentals are now complete.

Next:

```text
Computer Vision
↓
Images as tensors
↓
[channel, height, width]
↓
RGB channels and pixel values
↓
torchvision
↓
image datasets and transforms
↓
CNNs
```

The goal is to reuse the PyTorch foundations in this repository for real image-learning workflows.

---

## Purpose of This Repository

This is a **learning repository**, not a production ML system.

The emphasis is on understanding:

- why each PyTorch abstraction exists
- how tensor shapes flow through a model
- how data moves through a training pipeline
- how training and inference differ
- how to structure code so models can eventually be deployed reliably

The long-term goal is to build toward practical **AI engineering**, with later work in Computer Vision and more advanced deep-learning systems.
