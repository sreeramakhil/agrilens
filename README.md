# AgriLens

AgriLens is a Python toolkit for extracting actionable insights from agricultural imagery and related data. It provides utilities for data ingestion, preprocessing, model training, inference, evaluation, and deployment focused on remote sensing and field-level agricultural analytics (e.g., crop health monitoring, disease detection, yield estimation, and field segmentation).

## Table of contents
- [Project overview](#project-overview)
- [Key features](#key-features)
- [Repository structure](#repository-structure)
- [Getting started](#getting-started)
  - [Requirements](#requirements)
  - [Installation](#installation)
  - [Environment setup (recommended)](#environment-setup-recommended)
- [Quickstart examples](#quickstart-examples)
  - [Run preprocessing pipeline](#run-preprocessing-pipeline)
  - [Train a model](#train-a-model)
  - [Run inference](#run-inference)
- [Data format and ingestion](#data-format-and-ingestion)
- [Modeling and training](#modeling-and-training)
  - [Training configuration](#training-configuration)
  - [Evaluating models](#evaluating-models)
- [Deployment](#deployment)
  - [Docker](#docker)
  - [REST API / Serving](#rest-api--serving)
- [Configuration & environment variables](#configuration--environment-variables)
- [Testing & CI](#testing--ci)
- [Development workflow](#development-workflow)
- [Contributing](#contributing)
- [Roadmap](#roadmap)
- [Troubleshooting & FAQ](#troubleshooting--faq)
- [Citing & license](#citing--license)
- [Contact & acknowledgements](#contact--acknowledgements)

---

## Project overview
AgriLens centralizes common tasks for agricultural computer-vision and remote-sensing workflows:
- Preprocessing satellite / drone imagery (GeoTIFF, multispectral) and metadata
- Building datasets and augmentations for segmentation/classification/regression
- Training and evaluating deep learning models (PyTorch / TensorFlow compatible)
- Lightweight inference pipelines for field-level predictions
- Tools for model explainability, monitoring, and deployment

Intended users:
- Data scientists working on precision agriculture
- SRE/ML engineers deploying models for agritech products
- Researchers needing reproducible pipelines for remote sensing experiments

## Key features
- Modular preprocessing pipelines (tiling, normalization, cloud masking)
- Dataset utilities for common agricultural label formats (shapefiles, geojson, CSV)
- Config-driven training (YAML/JSON) with checkpointing and logging
- Evaluation metrics for segmentation and regression (IoU, F1, MAE, RMSE)
- Example deployment via Docker and a minimal REST API
- Reproducible experiments via deterministic seeds and environment exports

## Repository structure
A suggested/typical layout — update this section to match the repo exactly.

- README.md — (this file)
- LICENSE
- pyproject.toml or setup.cfg / requirements.txt — package metadata & dependencies
- src/agrilens/ — main Python package
  - data/ — data ingestion & augmentation utilities
  - preprocessing/ — image processing pipelines
  - models/ — model architectures and wrappers
  - training/ — training loops, schedulers, training CLI
  - inference/ — prediction and postprocessing utilities
  - api/ — lightweight REST service for predictions
  - utils/ — logging, config, IO helpers
- notebooks/ — Jupyter notebooks for exploration and demos
- experiments/ — saved configs, checkpoints, and logs
- docker/ — Dockerfiles and Kubernetes manifests
- tests/ — unit and integration tests

## Getting started

### Requirements
- Python 3.8+ (3.10 recommended)
- Recommended system requirements for model training:
  - CUDA-enabled GPU for deep learning (optional for CPU-only experiments)
  - >=16 GB RAM (varies with dataset size)
- Tools:
  - git
  - docker (optional for containerized runs)

### Installation

Clone the repository:
```bash
git clone https://github.com/sreeramakhil/agrilens.git
cd agrilens
```

Create and activate a virtual environment (recommended):
```bash
python -m venv .venv
source .venv/bin/activate   # macOS / Linux
.venv\Scripts\activate      # Windows (PowerShell)
```

Install dependencies:
- If using pip and requirements.txt:
```bash
pip install -r requirements.txt
```
- If using pip + project editable install:
```bash
pip install -e .
```

If you use conda:
```bash
conda create -n agrilens python=3.10
conda activate agrilens
pip install -r requirements.txt
```

### Environment setup (recommended)
Copy the sample environment and fill values:
```bash
cp .env.sample .env
# edit .env: set S3 buckets, local data paths, API keys, etc.
```

## Quickstart examples

These commands assume you have dataset files under DATA_PATH and that environment variables are configured.

Run preprocessing pipeline:
```bash
python -m agrilens.preprocessing.run \
  --config configs/preprocessing/tiling.yml \
  --input-dir /data/raw \
  --output-dir /data/processed
```

Train a model (example):
```bash
python -m agrilens.training.train \
  --config configs/training/segmentation_resnet50.yml \
  --data-dir /data/processed \
  --output-dir experiments/run-001
```

Run inference on a single file:
```bash
python -m agrilens.inference.predict \
  --model experiments/run-001/checkpoint.pth \
  --input /data/processed/tiles/tile_001.tif \
  --output /results/tile_001_pred.tif
```

Run local server for real-time predictions:
```bash
# start the service
python -m agrilens.api.server --port 8080 --model experiments/run-001/checkpoint.pth

# POST image to endpoint
curl -X POST -F "file=@tile_001.tif" http://localhost:8080/predict
```

## Data format and ingestion
AgriLens expects imagery and labels in common geospatial formats. Typical setups:

- Imagery: GeoTIFF (single or multi-band), JPEG/PNG (orthomosaic), HDF5, or numpy arrays
- Labels:
  - Segmentation masks as GeoTIFF or PNG (aligned with tiles)
  - Per-field CSV: field_id, geometry (WKT/GeoJSON), label(s), timestamp
  - Shapefiles / GeoJSON for polygons (field boundaries, annotations)

Best practices:
- Keep coordinate reference systems (CRS) consistent across imagery and labels.
- Use tiling to break large images into manageable patches.
- Keep example metadata alongside images (JSON or CSV) with:
  - acquisition date, sensor/bands, resolution, cloud cover estimate

Data ingestion example (pseudo-code):
```python
from agrilens.data import GeoTiffLoader, ShapefileLabelLoader

img = GeoTiffLoader.open('tile_001.tif')  # returns numpy array + profile
labels = ShapefileLabelLoader.load('labels.shp')  # returns vector polygons
```

## Modeling and training

Supported model types:
- Semantic segmentation (U-Net, DeepLab variants)
- Classification (ResNet, EfficientNet)
- Regression (yield estimation models)
- Multi-modal architectures (image + tabular farm metadata)

Training entrypoint:
- agrilens.training.train — config-driven; supports:
  - mixed precision (AMP)
  - distributed training (torch.distributed)
  - checkpointing and resume
  - callbacks (early stopping, model checkpoint, logging to TensorBoard / Weights & Biases)

Example training config (YAML snippet):
```yaml
model:
  name: unet_resnet34
  pretrained: true
training:
  batch_size: 8
  epochs: 60
  optimizer:
    name: adamw
    lr: 1e-4
scheduler:
  name: CosineAnnealingLR
data:
  num_workers: 6
  augmentations:
    - random_flip
    - random_rotation
```

### Training configuration
- Put experiment configurations under `configs/training/`.
- Use deterministic seeds for reproducibility.
- Log training progress to both console and TensorBoard/Weighs & Biases.

### Evaluating models
- Use relevant metrics: IoU / Dice / F1 for segmentation, RMSE/MAE for regression.
- Use geo-aware evaluation if aggregating predictions to field-level (e.g., average predicted NDVI per polygon and compare with ground truth).
- Example evaluation CLI:
```bash
python -m agrilens.training.evaluate \
  --config configs/eval/segmentation_eval.yml \
  --predictions /results/preds \
  --ground-truth /data/processed/masks
```

## Deployment

### Docker
Provided a Dockerfile for containerized inference/training:
```bash
# build the image
docker build -t agrilens:latest -f docker/Dockerfile .

# run inference container (mount model and data)
docker run --rm -v /models:/models -v /data:/data agrilens:latest \
  python -m agrilens.inference.predict --model /models/checkpoint.pth --input /data/tile_001.tif
```

### REST API / Serving
- A minimal FastAPI/Flask server is included in `src/agrilens/api`.
- Endpoints:
  - POST /predict: accepts image file and returns predictions (geojson or raster)
  - GET /health: health check
- Production suggestions:
  - Serve behind a WSGI server (gunicorn + uvicorn workers) or use a model server (TorchServe, Triton)
  - Add authentication, rate limiting, and request validation
  - Log requests and round-trip times for monitoring

## Configuration & environment variables
Centralize configurable parameters in a config file or environment variables. Example `.env` keys:
- DATA_ROOT=/data
- MODEL_DIR=/models
- S3_BUCKET=my-bucket
- WANDB_API_KEY=xxxx
- CUDA_VISIBLE_DEVICES=0

Load variables using python-dotenv or a config manager in `src/agrilens/utils/config.py`.

## Testing & CI
- Unit tests in `tests/` run with pytest.
- Recommended CI steps:
  - Run linters (flake8 / ruff)
  - Run formatting (black)
  - Run unit tests (pytest) with coverage
  - Build and test Docker image (integration tests)
  - Run lightweight smoke tests on example data (small fixtures)

Example CI command:
```bash
pytest --maxfail=1 --disable-warnings -q
```

## Development workflow
- Fork or branch from main
- Use feature branches: feature/<short-description>
- Run pre-commit hooks (black, ruff)
- Keep PRs focused and include:
  - Description of change
  - Screenshots / metrics if applicable
  - Test plan

## Contributing
Contributions are welcome. Please follow:
1. Open an issue to discuss large changes or features.
2. Create branches from `main`.
3. Add tests for new features / bug fixes.
4. Submit a PR and link related issue(s).


Built with ❤️ by Akhil
