# AgriLens

<div align="center">

**An Enterprise-Grade Deep Learning Framework for Precision Agriculture & Remote Sensing**

[![Python](https://img.shields.io/badge/Python-3.8+-3776ab?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Compatible-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed?style=flat-square&logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)

</div>

---

## 📋 Table of Contents

- [Executive Summary](#executive-summary)
- [Key Features](#key-features)
- [Technical Stack](#technical-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
  - [System Requirements](#system-requirements)
  - [Installation](#installation)
  - [First Run](#first-run)
- [Core Capabilities](#core-capabilities)
  - [Data Engineering Pipeline](#data-engineering-pipeline)
  - [Model Development & Training](#model-development--training)
  - [Inference & Deployment](#inference--deployment)
- [Advanced Features](#advanced-features)
  - [Configuration Management](#configuration-management)
  - [Performance Monitoring](#performance-monitoring)
  - [Production Deployment](#production-deployment)
- [Development & Contribution](#development--contribution)
- [Roadmap](#roadmap)
- [FAQ & Troubleshooting](#faq--troubleshooting)
- [License & Attribution](#license--attribution)

---

## 🎯 Executive Summary

**AgriLens** is a production-ready Python framework designed to accelerate machine learning workflows in precision agriculture and remote sensing. It abstracts away infrastructure complexity while maintaining flexibility for research and deployment.

### Impact:
- **74% Python-based** backend for maximum flexibility and industry compatibility
- **End-to-end automation** from raw satellite/drone imagery to field-level predictions
- **Modular architecture** enabling seamless integration into existing agritech platforms
- **Production-grade** with containerization, API serving, and monitoring support

### Ideal For:
- **ML Engineers & Data Scientists** building climate-tech and precision agriculture solutions
- **Researchers** conducting reproducible computer vision experiments on geospatial data
- **SRE/DevOps Teams** deploying scalable inference pipelines
- **Agritech Companies** needing rapid prototyping and production deployment

---

## ⚡ Key Features

### 🔄 **Data Processing Pipeline**
- Multi-format imagery ingestion (GeoTIFF, multispectral, orthomosaic)
- Geospatial-aware preprocessing: cloud masking, normalization, tiling
- Augmentation library optimized for satellite/drone imagery
- Metadata management and tracking throughout pipeline

### 🧠 **ML Model Support**
- Semantic segmentation (U-Net, DeepLab, Mask R-CNN variants)
- Classification networks (ResNet, EfficientNet, Vision Transformers)
- Regression models (yield prediction, pest pressure estimation)
- Multi-modal architectures (image + tabular farm metadata fusion)

### ⚙️ **Training Infrastructure**
- Configuration-driven training (YAML/JSON) for reproducibility
- Distributed training support (PyTorch DistributedDataParallel)
- Mixed precision training (Automatic Mixed Precision)
- Integration with TensorBoard and Weights & Biases for experiment tracking
- Checkpoint management with automated recovery

### 🚀 **Inference & Serving**
- Lightweight inference pipelines with batching support
- REST API (FastAPI) for real-time predictions
- Docker containerization for portable deployment
- Geo-aware postprocessing and aggregation

### 📊 **Evaluation & Metrics**
- Segmentation: IoU, Dice coefficient, F1-score, precision/recall
- Regression: MAE, RMSE, R² score
- Field-level geo-spatial aggregation
- Comprehensive evaluation reports

### 🔍 **Model Explainability**
- Attention map visualization
- Feature importance analysis
- SHAP value integration (roadmap)
- Prediction uncertainty quantification

---

## 🛠️ Technical Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Core Language** | Python 3.8+ | 74% of codebase |
| **Deep Learning** | PyTorch / TensorFlow | Model training & inference |
| **Data Processing** | NumPy, Rasterio, GeoPandas | Geospatial I/O |
| **Web Framework** | FastAPI | REST API server |
| **Containerization** | Docker, Docker Compose | Deployment & reproducibility |
| **Configuration** | YAML/JSON | Infrastructure as code |
| **Styling** | CSS (18.7%) | Web UI components |
| **Infrastructure** | PowerShell, Batchfile | Cross-platform scripting |

---

## 📁 Repository Structure

```
agrilens/
├── src/agrilens/
│   ├── data/                     # Data loading & augmentation
│   │   ├── loaders.py            # GeoTIFF, shapefile, CSV readers
│   │   ├── augmentation.py       # Spatial transformations
│   │   └── dataset.py            # PyTorch Dataset classes
│   ├── preprocessing/            # Image processing pipelines
│   │   ├── tiling.py             # Large image tiling
│   │   ├── normalization.py      # Band normalization (NDVI, etc.)
│   │   └── cloud_masking.py      # Sentinel-2 cloud detection
│   ├── models/                   # Neural network architectures
│   │   ├── segmentation.py       # U-Net, DeepLab implementations
│   │   ├── classification.py     # ResNet, EfficientNet wrappers
│   │   └── multimodal.py         # Image + tabular fusion models
│   ├── training/                 # Training orchestration
│   │   ├── trainer.py            # Main training loop
│   │   ├── schedulers.py         # Learning rate schedules
│   │   └── callbacks.py          # Early stopping, checkpointing
│   ├── inference/                # Prediction pipelines
│   │   ├── predictor.py          # Batch & single-image inference
│   │   ├── postprocess.py        # Result aggregation
│   │   └── metrics.py            # Evaluation metrics
│   ├── api/                      # REST API service
│   │   ├── server.py             # FastAPI application
│   │   ├── schemas.py            # Request/response models
│   │   └── middleware.py         # Authentication, logging
│   └── utils/                    # Shared utilities
│       ├── config.py             # Configuration management
│       ├── logging.py            # Structured logging
│       └── io.py                 # File I/O helpers
├── configs/                      # Training & preprocessing configs
│   ├── training/                 # Model training configurations
│   └── preprocessing/            # Data pipeline configurations
├── notebooks/                    # Jupyter notebooks (exploration & demos)
├── tests/                        # Unit & integration tests
├── docker/                       # Dockerfile & Kubernetes manifests
├── experiments/                  # Saved checkpoints & logs (gitignored)
├── requirements.txt              # Python dependencies
├── pyproject.toml               # Package metadata
├── Dockerfile                    # Container specification
└── README.md                     # This file
```

---

## 🚀 Quick Start

### System Requirements

| Component | Recommendation | Minimum |
|-----------|---|---|
| **Python** | 3.10+ | 3.8+ |
| **RAM** | 32 GB (GPU training) | 16 GB |
| **GPU** | NVIDIA with CUDA 11.8+ | Optional (CPU fallback) |
| **Storage** | 100+ GB | Varies by dataset |
| **OS** | Linux / macOS | Windows (PowerShell support included) |

### Installation

**1. Clone Repository**
```bash
git clone https://github.com/sreeramakhil/agrilens.git
cd agrilens
```

**2. Create Virtual Environment**
```bash
# Using venv (recommended)
python3 -m venv .venv
source .venv/bin/activate          # macOS / Linux
# OR
.venv\Scripts\activate              # Windows (PowerShell)
```

**3. Install Dependencies**
```bash
# Standard installation
pip install -r requirements.txt

# OR editable install (for development)
pip install -e ".[dev]"
```

**4. Configure Environment**
```bash
cp .env.sample .env
# Edit .env with your settings:
# - DATA_ROOT: path to dataset directory
# - S3_BUCKET: optional AWS S3 bucket
# - WANDB_API_KEY: Weights & Biases integration
# - CUDA_VISIBLE_DEVICES: GPU selection
```

### First Run

**Preprocess Sample Data:**
```bash
python -m agrilens.preprocessing.run \
  --config configs/preprocessing/tiling.yml \
  --input-dir ./data/raw \
  --output-dir ./data/processed
```

**Train Model:**
```bash
python -m agrilens.training.train \
  --config configs/training/segmentation_resnet50.yml \
  --data-dir ./data/processed \
  --output-dir ./experiments/baseline-v1
```

**Run Inference:**
```bash
python -m agrilens.inference.predict \
  --model ./experiments/baseline-v1/checkpoint.pth \
  --input ./data/processed/test_tile.tif \
  --output ./results/prediction.tif
```

**Start REST API:**
```bash
python -m agrilens.api.server \
  --port 8080 \
  --model ./experiments/baseline-v1/checkpoint.pth
```

---

## 🔧 Core Capabilities

### Data Engineering Pipeline

AgriLens handles multi-format geospatial data with production-grade robustness:

**Supported Input Formats:**
- **Imagery**: GeoTIFF (single/multi-band), Sentinel-2 L2A, orthomosaics (JPEG/PNG), HDF5
- **Labels**: Segmentation masks (GeoTIFF/PNG), shapefiles, GeoJSON, CSV with WKT geometries
- **Metadata**: JSON manifests with acquisition date, sensor specs, resolution, cloud cover

**Key Operations:**
```python
from agrilens.data import GeoTiffLoader, ShapefileLabelLoader

# Load multispectral imagery
imagery, profile = GeoTiffLoader.open('sentinel2_tile.tif')  # Returns (H, W, Bands) + geoprofile

# Load vector labels
polygons = ShapefileLabelLoader.load('field_boundaries.shp')  # Returns GeoDataFrame

# Automatic rasterization & alignment
mask = polygons.to_raster(imagery.shape, profile)
```

### Model Development & Training

**Supported Architectures:**
- Segmentation: U-Net (with ResNet/VGG backbone), DeepLab v3+, Mask R-CNN
- Classification: ResNet50/101, EfficientNet-B0 to B7, Vision Transformer (ViT)
- Regression: Custom CNN, Transformer-based yield models
- Multi-modal: Concatenation-based fusion, cross-attention mechanisms

**Training Configuration (YAML):**
```yaml
model:
  name: unet_resnet50
  pretrained: true
  encoder_weights: imagenet

training:
  batch_size: 16
  epochs: 100
  optimizer:
    name: adamw
    lr: 1e-4
    weight_decay: 1e-5

scheduler:
  name: CosineAnnealingLR
  t_max: 100
  eta_min: 1e-6

data:
  num_workers: 8
  pin_memory: true
  augmentations:
    - RandomHorizontalFlip
    - RandomVerticalFlip
    - RandomRotation(90)
    - GaussNoise(0.01)
```

**Key Features:**
- Mixed precision training (Automatic Mixed Precision) for 2x speedup
- Distributed training across multiple GPUs
- Automated checkpointing and recovery
- Integration with TensorBoard & Weights & Biases
- Deterministic seeding for reproducibility

### Inference & Deployment

**Single-Image Inference:**
```bash
python -m agrilens.inference.predict \
  --model checkpoint.pth \
  --input tile.tif \
  --output prediction.tif \
  --device cuda
```

**Batch Processing:**
```bash
python -m agrilens.inference.batch \
  --model checkpoint.pth \
  --input-dir tiles/ \
  --output-dir predictions/ \
  --batch-size 32
```

**REST API Endpoints:**
```
POST /predict       - Upload image, get predictions (GeoJSON or GeoTIFF)
POST /predict-batch - Batch processing job submission
GET /health         - Service health check
GET /status/{job_id} - Monitor batch job progress
```

---

## 🎛️ Advanced Features

### Configuration Management

Centralized configuration through environment variables and config files:
```bash
# .env file
DATA_ROOT=/mnt/data
MODEL_DIR=/mnt/models
S3_BUCKET=agritech-models
WANDB_PROJECT=agrilens-prod
CUDA_VISIBLE_DEVICES=0,1,2,3
LOG_LEVEL=INFO
```

### Performance Monitoring

Built-in observability:
- **Metrics**: Training loss, validation metrics, inference latency
- **Logging**: Structured JSON logs, traceable through request IDs
- **Profiling**: Memory usage, compute utilization per batch
- **Dashboards**: TensorBoard integration for live monitoring

### Production Deployment

**Docker Containerization:**
```bash
# Build production image
docker build -t agrilens:prod -f docker/Dockerfile.prod .

# Run inference container
docker run --gpus all \
  -v /models:/models \
  -v /data:/data \
  -p 8080:8080 \
  agrilens:prod \
  python -m agrilens.api.server --port 8080

# Kubernetes deployment (optional)
kubectl apply -f docker/k8s/deployment.yaml
```

**Scaling Considerations:**
- Horizontal scaling via Docker Swarm or Kubernetes
- Load balancing with Nginx/HAProxy
- Model versioning and A/B testing support
- Request queuing and backpressure handling

---

## 👨‍💻 Development & Contribution

### Development Workflow

1. **Fork** repository and create feature branch:
   ```bash
   git checkout -b feature/geospatial-metrics
   ```

2. **Setup development environment:**
   ```bash
   pip install -e ".[dev]"
   pre-commit install
   ```

3. **Write tests:**
   ```bash
   pytest tests/ -v --cov=agrilens
   ```

4. **Code quality:**
   ```bash
   black src/
   ruff check src/ --fix
   mypy src/ --strict
   ```

5. **Submit pull request** with:
   - Description of changes
   - Related issue links
   - Test coverage
   - Performance metrics (if applicable)

### Contribution Guidelines

- Follow [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide
- Add docstrings (Google format) for all public functions
- Include unit tests (>80% coverage target)
- Update documentation for user-facing changes
- Reference issue numbers in commit messages

---

## 🗺️ Roadmap

### Q2 2026
- [ ] Vision Transformer (ViT) architecture support
- [ ] Multi-temporal sequence processing
- [ ] SHAP-based model explainability

### Q3 2026
- [ ] Federated learning support for privacy-preserving training
- [ ] Real-time streaming inference pipeline
- [ ] Web-based annotation tool

### Q4 2026
- [ ] MLOps integration (MLflow, DVC)
- [ ] AutoML configuration optimization
- [ ] Commercial SaaS platform

---

## ❓ FAQ & Troubleshooting

**Q: Can I train on CPU?**  
A: Yes, but GPU is highly recommended for production models. Training on CPU will be 10-50x slower.

**Q: What's the minimum dataset size?**  
A: Start with 100+ labeled images for baseline models. Production models typically require 1000+ samples.

**Q: How do I handle class imbalance?**  
A: Use weighted loss functions (WeightedBCELoss) or focal loss. See `configs/training/imbalanced.yml` for example.

**Q: Can I use my own model architecture?**  
A: Yes! Inherit from `BaseSegmentationModel` in `src/agrilens/models/base.py`.

**Q: How do I deploy to production?**  
A: Use provided Docker setup or Kubernetes manifests. See `docker/` directory for examples.

---

## 📄 License & Attribution

**License:** MIT License (see [LICENSE](LICENSE) file)

**Author:** [Akhil Sreerama](https://github.com/sreeramakhil)

**Built with:** ❤️ for the precision agriculture community

### Acknowledgments

- Inspired by best practices in computer vision and geospatial ML
- Community contributions welcome—please see [CONTRIBUTING.md](CONTRIBUTING.md)

---

<div align="center">

**Questions?** Open an [issue](https://github.com/sreeramakhil/agrilens/issues) or contact the maintainers.

</div>
