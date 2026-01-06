# PyTorch Classifiers (CIFAR-100)

This project trains and evaluates multiple image classifiers on **CIFAR-100** using **PyTorch**, and compares:
- a fully-connected **MLP** (≤ 5M parameters),
- a compact **CNN** (≤ 5M parameters),
- a **CNN ablation** (BatchNorm removed),
- a **pretrained ViT-Base** model from Hugging Face evaluated on the CIFAR-100 test set.

All work is contained in the notebook: `pytorch_classifiers.ipynb`.

---

## Results

| Model | Parameters (M) | Test Accuracy |
|---|---:|---:|
| MLP | 3.83 | 26.9% |
| CNN | 1.17 | 65.8% |
| CNN (No BN) | 1.17 | 55.5% |
| ViT-Base (Pretrained on CIFAR-100) | 85.9 | 92.6% |

---

## Repository Contents

- `pytorch_classifiers.ipynb` — end-to-end pipeline:
  - data loading (CIFAR-100)
  - model definitions (MLP, CNN, ablation)
  - training loops and evaluation
  - pretrained Hugging Face model evaluation
  - comparison + discussion

---

## Requirements

- Python 3.9+
- PyTorch + torchvision
- transformers (Hugging Face)
- numpy
- Jupyter

---

## Setup

Create and activate an environment (recommended):

```bash
python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

Install dependencies:

```bash
pip install --upgrade pip
pip install torch torchvision transformers numpy jupyter
```

> If you want GPU support, install the correct PyTorch build for your CUDA version from the official PyTorch installation instructions.

---

## Run

```bash
jupyter lab
```

Open `pytorch_classifiers.ipynb` and run the cells top-to-bottom.

---

## What the notebook does

### Models
- **MLP (≤ 5M params):** Fully-connected network on flattened 32×32×3 input.
- **CNN (≤ 5M params):** Small 3×3 conv blocks with pooling/downsampling to keep parameters low.
- **CNN (No BN):** Same CNN architecture but **BatchNorm removed** for ablation.
- **Pretrained ViT-Base:** Loaded via Hugging Face and evaluated on CIFAR-100 test set.

### Reporting
For each model, the notebook prints:
- `Number of model parameters is: ...`
- `Test accuracy of model: XX.X%`

---

## Notes

- CIFAR-100 is downloaded automatically through `torchvision.datasets` on first run.
- Training time depends on hardware (CPU vs GPU).
- Exact hyperparameters and transforms are documented inline in the notebook.

---

## License

For coursework/educational use. Add a LICENSE file if you plan to distribute publicly.
