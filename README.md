# Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification (CVSL)

[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](requirements.txt)
[![Status](https://img.shields.io/badge/status-research%2Fexperimental-orange)](#status)
[![Publication](https://img.shields.io/badge/PDF-view-orange)\](https://drive.google.com/file/d/1NAmaJF7buBkKZmAcY181DCdm3XpCNvRN/view?usp=sharing)

This repository contains research code for **CVSL (Contrastive Viewpoint-aware Shape Learning)**, a Long-term Person Re-Identification (LRe-ID) method that improves robustness to **clothing changes** and **viewpoint variations** by combining appearance cues with texture-invariant body shape cues.

## Paper summary (WACV 2024)

### Problem
Classic Re-ID methods rely on appearance. In long-term scenarios this breaks down when:

- the same person changes clothes / hairstyle, or their face is occluded;
- different people wear similar clothes.

In addition, viewpoint shifts (front/side/back) can cause both texture and shape embeddings to drift, creating false matches.

### Method: CVSL

CVSL has two feature branches and learns them with viewpoint-aware objectives.

1) Relational Shape Embedding (RSE) branch

- Extracts **2D pose keypoints** and encodes them as a graph.
- Uses a **refinement MLP** to lift raw joint coordinates to a higher-dimensional space.
- Uses a **Graph Attention Network (GAT)** over the skeleton graph to capture local part relations and higher-order shape structure.
- Produces a global **shape embedding** via global pooling.

2) Texture (appearance) branch

- Uses a CNN backbone (ResNet-50 in the paper) to extract appearance features.
- Uses clothing-aware objectives to discourage over-reliance on clothing texture.

3) Contrastive Viewpoint-aware Losses (CVL)

- **Shape CVL**: positive pairs are the same identity across different viewpoints; negatives are different identities under the same viewpoint.
- **Appearance CVL**: encourages cross-view consistency and includes a hard-mined component to handle look-alike clothing cases.

4) Adaptive Fusion Module (AFM)

Instead of naive concatenation, AFM projects shape/appearance features to a shared space and learns adaptive weights:

$$f = w^s \odot f^s + w^a \odot f^a$$

### Results (cloth-changing setting)

Reported in the paper:

- LTCC: Rank-1 44.5, mAP 21.3
- PRCC: Rank-1 57.5, mAP 56.9

## Repository overview

Key code locations:

- Model: `src/models/cvsl_reid.py` (CVSL-style appearance + optional shape + fusion)
- Shape encoder: `src/models/modules/shape_embedding.py`
- Datasets / samplers: `src/datasets/`
- Pose extraction (HRNet): `tools/get_pose.py`
- Orientation extraction (HOE): `tools/get_orientation.py`
- Example training loop (orientation-guided triplets): `tools/train_orientation_contrastive.py`
- Evaluation script (baseline pipeline): `test.py`

Note: some scripts are "research/experimental" and assume specific file names (e.g. `external_data/ltcc/pose_train.json`) or contain hard-coded defaults. The sections below document the paths and formats the code expects.

## Installation

### Python

The repo is Python + PyTorch. Install dependencies with:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Docker

Docker support is provided via `Dockerfile` + `docker-compose.yaml`:

```bash
docker-compose up -d
docker exec -it CVSL_ReID bash
```

For detailed environment notes (including NVIDIA Container Toolkit), see docs/INSTALL.md.

## Data: expected formats

### Datasets

The dataset loaders in `src/datasets/` support LTCC and PRCC.

- LTCC expects a directory containing `train/`, `query/`, and `test/` images.
- PRCC expects a directory containing `rgb/train/`, `rgb/val/`, and `rgb/test/`.

### Metadata JSON (external_data)

Several parts of the code rely on JSON metadata entries of the form:

```json
{
  "img_path": "/abs/path/to/image.png",
  "p_id": 73,
  "cam_id": 10,
  "clothes_id": 235,
  "orientation": 180,
  "pose_landmarks": [[x, y, score], ...]
}
```

This repo includes example files under `external_data/`.

Important: these JSONs may contain **absolute paths** from the original author machine. If your dataset lives elsewhere, regenerate them (recommended).

## Preprocessing (pose + orientation)

### 1) Orientation (HOE)

Create `external_data/<dataset-name>/<split>.json` with orientation labels:

```bash
PYTHONPATH=. python tools/get_orientation.py \
  --dataset-name ltcc \
  --dataset /path/to/LTCC_ReID \
  --target-set train \
  --device cuda \
  --batch-size 32
```

This produces `external_data/ltcc/train.json`.

### 2) Pose landmarks (HRNet)

Augment metadata with 2D pose landmarks:

```bash
PYTHONPATH=. python tools/get_pose.py \
  --metadata external_data/ltcc/train.json \
  --dataset-name ltcc \
  --target-set pose_train \
  --batch-size 32
```

This produces `external_data/ltcc/pose_train.json` which is the default expected by `tools/train_orientation_contrastive.py`.

## Training

### Orientation-guided contrastive (example)

The simplest runnable example is:

```bash
PYTHONPATH=. python tools/train_orientation_contrastive.py --epochs 10 --learning-rate 1e-3
```

Options:

- `--epochs` (default: 10)
- `--learning-rate` (default: 1e-3)
- `--log-every-n-epochs` (default: 1)
- `--ckpt` (load `checkpoints/model.ckpt`)

This script expects `external_data/ltcc/pose_train.json` and writes a PyTorch state dict to `checkpoints/model.ckpt`.

## Evaluation

The evaluation entrypoint is `test.py`. It uses settings in `config.py` (dataset paths, dataset name, output folders, etc.). Update `BASIC_CONFIG.DATASET_PATH` / `BASIC_CONFIG.DATASET_NAME` and then run:

```bash
PYTHONPATH=. python test.py
```

## Citation

If you use this work in your research, please cite the WACV 2024 paper:

- Vuong D. Nguyen, Khadija Khaldi, Dung Nguyen, Pranav Mantini, Shishir Shah. "Contrastive Viewpoint-aware Shape Learning for Long-term Person Re-Identification." WACV 2024.

BibTeX: (add the official BibTeX from the CVF page)

## 📄 Citation

If you use this code in your research, please cite:

```bibtex
@InProceedings{Nguyen_2024_WACV,
    author    = {Vuong D. Nguyen, Khadija Khaldi, Dung Nguyen, Pranav Mantini, Shishir Shah},
    title     = {Contrastive Viewpoint-Aware Shape Learning for Long-Term Person Re-Identification},
    booktitle = {Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision (WACV)},
    month     = {January},
    year      = {2024},
    url       = {https://openaccess.thecvf.com/content/WACV2024/html/Nguyen_Contrastive_Viewpoint-Aware_Shape_Learning_for_Long-Term_Person_Re-Identification_WACV_2024_paper.html}
}
```

**Paper Link**: [WACV 2024 Open Access](https://openaccess.thecvf.com/content/WACV2024/html/Nguyen_Contrastive_Viewpoint-Aware_Shape_Learning_for_Long-Term_Person_Re-Identification_WACV_2024_paper.html)

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📞 Support

For questions and issues:
1. Check the [Issues](../../issues) page
2. Review the documentation in `docs/`
3. Contact the authors

---

**Keywords**: Person Re-identification, Long-term ReID, Cloth-changing, Pose estimation, Graph Neural Networks, Contrastive Learning
