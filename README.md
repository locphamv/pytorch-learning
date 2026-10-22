# PyTorch Learning

A hands-on repository for learning **PyTorch, deep learning fundamentals, and computer vision** step by step.

This repo documents my progression from tensor basics and tabular pipelines to building and evaluating **Convolutional Neural Networks (CNNs)** and **transfer learning workflows** with:

- custom `Dataset` and `DataLoader`
- model training, validation, and evaluation loops
- Apple MPS, CUDA, and CPU device support
- binary and multiclass classification
- model persistence with `state_dict`
- preprocessing transforms and pipeline serialization
- train / validation / test discipline to prevent data leakage
- best-checkpoint selection
- computer vision with `torchvision` (v2 transforms, image tensors)
- CNN architecture design from scratch
- data augmentation strategies
- transfer learning and fine-tuning with pretrained ResNet18
- systematic post-training error analysis (confusion matrices, per-class metrics, high-confidence error visualization)

---

## Learning Roadmap

```text
PyTorch fundamentals
↓
Tensors, shapes, autograd
↓
nn.Module, losses, optimizers
↓
Neural networks (MLPs)
↓
Dataset / DataLoader & batching
↓
Train / validation / test splits
↓
Device-aware training (CPU / MPS / CUDA)
↓
Model save / load & state_dict
↓
Multiclass classification
↓
Custom CSV datasets & tabular preprocessing
↓
End-to-end training pipeline & packaged checkpoints
↓
Computer Vision: Images as tensors [C, H, W]
↓
Convolutions & feature extraction
↓
Convolutional Neural Networks (CNNs)
↓
CIFAR-10 benchmarking
↓
Data augmentation (torchvision v2)
↓
Transfer learning & fine-tuning (ResNet18)
↓
Systematic error analysis & inspection
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
- broadcasting semantics
- multiple outputs
- `reshape`, `unsqueeze`, `squeeze`
- transpose and permute
- debugging tensor shapes

### 3. Autograd and Gradient Descent

- `requires_grad=True`
- computation graphs
- `loss.backward()`
- parameter gradients
- manual gradient descent
- clearing accumulated gradients (`optimizer.zero_grad()`)

### 4. `nn.Module`, Losses, and Optimizers

- custom PyTorch modules
- `nn.Linear`
- `forward()` execution
- `nn.MSELoss`
- `torch.optim.SGD`
- `model.train()` vs `model.eval()`
- `torch.inference_mode()`

### 5. Multi-Layer Neural Networks

- hidden layers
- ReLU activations
- nonlinear decision boundaries (XOR)
- `nn.Sequential`
- binary logits
- numerical stability with `BCEWithLogitsLoss`

### 6. Dataset and DataLoader

- `TensorDataset`
- batching and mini-batch gradient descent
- shuffling
- handling uneven final batches (`drop_last`)
- sample-weighted average loss calculation

### 7. Train / Validation / Test

- training loop vs validation loop
- held-out test evaluation
- overfitting and generalization
- preventing test-set leakage

### 8. Device Management

- CPU fallback
- Apple Silicon (MPS) acceleration
- NVIDIA GPU (CUDA) support
- moving models and batches to the same device
- portable inference and moving tensors back to CPU for NumPy/plotting

### 9. Model Persistence

- `model.state_dict()`
- `torch.save` and `torch.load`
- `load_state_dict`
- rebuilding a fresh model architecture
- verifying restored predictions

### 10. Multiclass Classification

- logits shaped `[batch_size, num_classes]`
- `CrossEntropyLoss` with unnormalized logits
- class IDs as `torch.long`
- `argmax(dim=1)` prediction
- softmax probabilities
- Adam optimizer

### 11. Custom Dataset from CSV

- subclassing `torch.utils.data.Dataset`
- implementing `__len__` and `__getitem__`
- reading tabular data with pandas
- input validation and feature contracts
- binary classification from external CSV data

### 12. Transforms and Preprocessing

- callable transform objects
- feature standardization (mean / std)
- training-only preprocessing statistics
- avoiding preprocessing data leakage
- transform composition
- reusable preprocessing pipelines

### 13. Complete Training Pipeline

- data validation
- deterministic data splitting
- computing training statistics
- modular train and evaluation functions
- validation loss tracking
- best-model checkpoint selection
- final test evaluation

### 14. Model + Preprocessing Checkpoint

- packaging architecture configuration, model weights, feature order, mean, standard deviation, and prediction threshold into a single `.pt` file
- self-contained raw-feature inference without external preprocessing code

---

### 15. Images as Tensors

- reading image files using `torchvision.io.decode_image`
- channel-first tensor format: `[channels, height, width]` (`[C, H, W]`)
- batch format: `[batch_size, channels, height, width]` (`[B, C, H, W]`)
- RGB color channels, pixel values, and spatial indexing
- modern torchvision transforms with `torchvision.transforms.v2`
- resizing and scaling `uint8` `[0, 255]` to `float32` `[0.0, 1.0]`

### 16. Convolutions from Scratch

- 2D sliding-window convolution algorithm implemented in pure PyTorch
- receptive fields and kernel weights
- spatial feature detectors (horizontal and vertical edge filters)
- output dimension formula: `(H - K + 1, W - K + 1)`
- padding and stride mechanics
- comparing manual implementations with `torch.nn.functional.conv2d` and `nn.Conv2d`

### 17. First Convolutional Neural Network (CNN)

- building a CNN module (`SmallCNN`)
- feature extractor backbone: `nn.Conv2d` -> `nn.ReLU` -> `nn.MaxPool2d`
- transition from 2D spatial feature maps to 1D vectors with `nn.Flatten`
- classifier head: `nn.Linear` layers with non-linear activation
- tracking tensor dimensions across convolution, pooling, flattening, and linear projections

### 18. Real-World Vision: CIFAR-10

- downloading and loading CIFAR-10 via `torchvision.datasets.CIFAR10`
- 10 natural image classes (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- image normalization: `v2.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))`
- splitting validation set from official training split with fixed seeds
- training loop, validation loss tracking, and checkpoint saving (`cifar10_cnn.pth`)

### 19. Data Augmentation

- preventing overfitting on vision models
- `torchvision.transforms.v2` augmentation primitives:
  - `v2.RandomCrop(size=32, padding=4, padding_mode="reflect")`
  - `v2.RandomHorizontalFlip(p=0.5)`
  - `v2.ColorJitter(brightness=0.1, contrast=0.1)`
- **critical rule**: apply data augmentation ONLY to the training loader; keep validation and testing strictly deterministic

### 20. Transfer Learning with ResNet18

- leveraging pretrained deep representations (`torchvision.models.resnet18`, `ResNet18_Weights.DEFAULT`)
- feature extraction: freezing backbone weights (`requires_grad = False`)
- adapting the classifier head: replacing `model.fc` with a new `nn.Linear(512, 10)`
- using official pretrained transform recipes (`weights.transforms()`)
- drastic reduction in training time and required training samples

### 21. Model Fine-Tuning

- difference between fixed feature extraction and fine-tuning
- saving and versioning specialized fine-tuned checkpoints (`models/resnet18_cifar10_finetuned.pth`)
- evaluation on subset validation sets for rapid iteration on Apple Silicon / CPU

### 22. Systematic Validation Error Analysis

- evaluating classification performance beyond summary accuracy
- computing confusion matrices using vectorized `torch.bincount` without third-party dependencies
- calculating per-class diagnostics: **Support**, **Recall**, **Precision**, and **F1-Score**
- detecting most frequent directed error pairs (e.g., `cat -> dog`, `automobile -> truck`)
- un-normalizing normalized image tensors back to `[0.0, 1.0]` RGB
- exporting full validation predictions to CSV (`outputs/lesson_22_<tag>_predictions.csv`)
- generating visual grids of highest-confidence mistakes (`outputs/lesson_22_<tag>_mistakes.png`) via `torchvision.utils.save_image`
- CLI support via `argparse` (`--checkpoint baseline` or `--checkpoint finetuned`)

---

## Project Structure

```text
pytorch-learning/
├── data/
│   ├── students.csv                       # Tabular training dataset
│   ├── sample.png                         # Sample image for tensor inspection
│   └── cifar-10-batches-py/               # Downloaded CIFAR-10 dataset
├── models/
│   ├── student_classifier_checkpoint.pt   # Packaged model + preprocessing pipeline
│   ├── cifar10_cnn.pth                    # Baseline CNN checkpoint
│   ├── cifar10_augmented.pth              # CNN trained with augmentation
│   ├── resnet18_cifar10.pth               # Pretrained ResNet18 baseline checkpoint
│   └── resnet18_cifar10_finetuned.pth     # Fine-tuned ResNet18 checkpoint
├── outputs/
│   ├── lesson_22_baseline_predictions.csv
│   ├── lesson_22_baseline_mistakes.png
│   ├── lesson_22_finetuned_predictions.csv
│   └── lesson_22_finetuned_mistakes.png
├── lesson_01_tensors.py
├── lesson_02_linear_algebra.py
├── lesson_03_autograd.py
├── lesson_04_first_model.py
├── lesson_04_exercise.py
├── lesson_05_neural_network.py
├── lesson_06_dataloader.py
├── lesson_06_exercise.py
├── lesson_07_evaluation.py
├── lesson_07_exercise.py
├── lesson_08_devices.py
├── lesson_09_save_load.py
├── lesson_10_multiclass.py
├── lesson_11_custom_dataset.py
├── lesson_12_transforms.py
├── lesson_13_training_pipeline.py
├── lesson_14_checkpoint.py
├── lesson_15_images_as_tensors.py
├── lesson_16_convolution.py
├── lesson_17_first_cnn.py
├── lesson_18_cifar10.py
├── lesson_19_augmentation.py
├── lesson_20_transfer_learning.py
├── lesson_21_finetuning.py
├── lesson_22_error_analysis.py
├── .gitignore
└── README.md
```

---

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/locphamv/pytorch-learning.git
cd pytorch-learning
```

### 2. Create and activate virtual environment

```bash
python -m venv .venv
```

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
python -m pip install torch torchvision numpy pandas matplotlib
```

---

## Running Lessons

### Tabular & Core PyTorch Fundamentals

```bash
# Tensor basics and training loop
python lesson_01_tensors.py
python lesson_04_first_model.py

# Multiclass and custom datasets
python lesson_10_multiclass.py
python lesson_11_custom_dataset.py

# End-to-end training pipeline and deployable checkpoint packaging
python lesson_13_training_pipeline.py
python lesson_14_checkpoint.py
```

### Computer Vision Pipeline

```bash
# Images as tensors & custom convolutions
python lesson_15_images_as_tensors.py
python lesson_16_convolution.py

# First CNN and CIFAR-10 classification
python lesson_17_first_cnn.py
python lesson_18_cifar10.py

# Data augmentation
python lesson_19_augmentation.py

# Transfer learning & fine-tuning with ResNet18
python lesson_20_transfer_learning.py
python lesson_21_finetuning.py

# Systematic error analysis (confusion matrix, metrics, error visualization)
python lesson_22_error_analysis.py --checkpoint baseline
python lesson_22_error_analysis.py --checkpoint finetuned
```

---

## Device Support

All model scripts include automatic hardware device detection:

```python
def get_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")
```

This guarantees acceleration on Apple Silicon (MPS) and NVIDIA GPUs (CUDA) while falling back seamlessly to CPU.

---

## Important Engineering Rules Practiced

- **Data hygiene**: Training, validation, and test sets have distinct responsibilities. The test set is strictly held out and never used for model selection or tuning.
- **Leakage prevention**: Preprocessing statistics (means, standard deviations, scaling parameters) must be fitted on **training data only**.
- **Augmentation discipline**: Data augmentations (random crop, flips, color jitters) must **only** be executed on the training data pipeline; evaluation transforms must remain strictly deterministic.
- **Device consistency**: Models and input tensors must reside on the identical device before execution.
- **Logits directly to loss**: Raw, unnormalized logits should be passed to `BCEWithLogitsLoss` or `CrossEntropyLoss` for numerical stability.
- **Pretrained transform contracts**: Pretrained architectures (like ResNet18) require their exact corresponding transform recipe (`weights.transforms()`).
- **Un-normalizing for human inspection**: When visualizing model errors or normalized images, invert the normalization step (`image * std + mean`) to inspect valid RGB images.
- **Accuracy is not enough**: Systematic error analysis requires confusion matrices, per-class recall/precision, and inspecting high-confidence false predictions.
- **Deployability**: A deployable model artifact includes architecture configuration, preprocessing contracts, and thresholds—not just weight tensors.

---

## Purpose of This Repository

This is a **learning repository**, not a production ML system.

The emphasis is on understanding:

- why each PyTorch and Deep Learning abstraction exists
- how tensor shapes flow through computation graphs and network layers
- how data moves cleanly through preprocessing, datasets, and data loaders
- how training, validation, and inference differ in practice
- how to build disciplined, reproducible, and debuggable AI pipelines
- how to structure code so models can be deployed and monitored reliably
