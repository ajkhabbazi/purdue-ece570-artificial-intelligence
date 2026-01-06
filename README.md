# purdue-ece570-artificial-intelligence

Coursework repository for **ECE 570 — Artificial Intelligence (Fall 2025)** at Purdue, taught by **Prof. David I. Inouye**.

- Course page: https://www.davidinouye.com/course/ece57000-fall-2025/
- Instructor: https://www.davidinouye.com/
- Author (assignments): **Arash Jalil Khabbazi**

This repo contains four self-contained assignments implemented primarily as Jupyter notebooks, each living under `assignments/` with its own local README and notebook.

## Repository layout

```text
.
├── assignments/
│   ├── classifiers/
│   │   ├── nasa_neos_classification.ipynb
│   │   └── README.md
│   ├── grpo-fine-tuning/
│   │   ├── grpo_fine_tuning.ipynb
│   │   └── README.md
│   ├── pytorch-classifiers/
│   │   ├── pytorch_classifiers.ipynb
│   │   └── README.md
│   └── transformers/
│       ├── transformers.ipynb
│       └── README.md
└── README.md
```

## Assignments

### 1) NASA Near-Earth Objects (NEOs) — Hazard Classification (`assignments/classifiers/`)
Binary classification on a NASA NEO dataset (Kaggle) to predict whether an asteroid is **potentially hazardous**. The notebook includes an end-to-end ML workflow (preprocessing, baselines, tuning, imbalance-aware evaluation) and compares models such as KNN, Logistic Regression, and an imbalance-aware XGBoost setup with threshold tuning.

- Notebook: `assignments/classifiers/nasa_neos_classification.ipynb`

### 2) GRPO Fine-Tuning on a Toy Addition Task (`assignments/grpo-fine-tuning/`)
A minimal RL fine-tuning example using **Group Relative Policy Optimization (GRPO)** to fine-tune a small causal LM (`distilgpt2`) on a toy one-digit addition task. The notebook demonstrates the full loop (prompting → sampling groups → reward → GRPO update → evaluation) and discusses reward design trade-offs.

- Notebook: `assignments/grpo-fine-tuning/grpo_fine_tuning.ipynb`

### 3) PyTorch Classifiers (CIFAR-100) (`assignments/pytorch-classifiers/`)
Training and evaluation of multiple image classifiers on **CIFAR-100** using **PyTorch**, including:
- a parameter-budgeted MLP,
- a compact CNN,
- a CNN ablation (BatchNorm removed),
- and evaluation of a pretrained ViT model from Hugging Face.

- Notebook: `assignments/pytorch-classifiers/pytorch_classifiers.ipynb`

### 4) Transformers (Decoder-Only) Language Model — Tiny Shakespeare (`assignments/transformers/`)
An educational implementation of a small **decoder-only Transformer** language model trained on **Tiny Shakespeare**, covering tokenization options, positional encodings, causal multi-head self-attention, training, and generation (temperature / top-k / top-p).

- Notebook: `assignments/transformers/transformers.ipynb`

## Getting started

### Prerequisites
- Python 3.9+ (3.10+ recommended)
- Jupyter (VS Code, Jupyter Lab, or Colab)

### Setup (local)
```bash
git clone https://github.com/ajkhabbazi/purdue-ece570-artificial-intelligence.git
cd purdue-ece570-artificial-intelligence

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1

pip install -U pip jupyter
```

### Install dependencies
Each assignment notebook is self-contained and may require different packages. See the `README.md` inside each assignment folder for the most accurate dependency list and notes.

As a starting point, these are commonly used across the assignments:
```bash
pip install -U numpy pandas scipy scikit-learn torch torchvision transformers tokenizers xgboost requests
```

## Running

### Option A: Jupyter (recommended for local runs)
```bash
jupyter lab
```
Then open any notebook under `assignments/**`.

### Option B: Google Colab
Upload the notebook to Colab (or open it from GitHub) and run cells top-to-bottom. Some notebooks may require credentials or dataset downloads (documented in the per-assignment README).

## Notes

- Notebooks are intended to be run **top-to-bottom**.
- Results can vary slightly depending on hardware (CPU/GPU), random seeds, and library versions.

## License

Coursework repository for educational use. If you plan to reuse or distribute this code publicly, add an explicit open-source license file.
