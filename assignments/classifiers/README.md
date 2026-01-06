# NASA Near-Earth Objects (NEOs) — Hazard Classification

Classify whether a near-Earth object (asteroid) is **potentially hazardous** using supervised machine learning on a public NASA NEO dataset from Kaggle.

> **Notebook:** `nasa_neos_classification.ipynb`  
> **Task:** Binary classification (`hazardous` vs. non-hazardous) with imbalanced classes.

---

## What’s in this project

- **End-to-end ML workflow** in a single notebook:
  - Dataset justification + problem framing
  - Data loading (Kaggle API when available) + preprocessing
  - Baseline + tuned models (KNN, Logistic Regression)
  - Feature pruning with Mutual Information (train-only)
  - “Choose your own adventure” model: **XGBoost** (imbalance-aware) + threshold tuning
- **Evaluation beyond accuracy**:
  - Balanced Accuracy, Precision, Recall, F1
  - ROC-AUC and PR-AUC (Average Precision)
  - Decision-threshold analysis for imbalanced classification

---

## Dataset

- **Source:** Kaggle dataset `sameepvani/nasa-nearest-earth-objects`
- **Size (as used in the notebook):** 90,836 rows × 10 columns
- **Target label:** `hazardous` (mapped to 0/1)
- **Columns dropped as identifiers/non-predictive (when present):** `id`, `name`, `sentry_object`, `orbiting_body`
- **Features used after preprocessing (5 total):**
  - `est_diameter_min`
  - `est_diameter_max`
  - `relative_velocity`
  - `miss_distance`
  - `absolute_magnitude`

---

## Results (test set)

Primary comparison (emphasis on imbalance-aware metrics):

| Model | Accuracy | Balanced Accuracy | F1 |
|---|---:|---:|---:|
| KNN (tuned) | 0.9169 | 0.6207 | 0.3720 |
| Logistic Regression (tuned) | 0.9022 | 0.5151 | 0.0643 |
| **XGBoost (imbalance-aware)** | 0.8612 | **0.8086** | **0.5104** |

**Takeaway:** XGBoost achieved the best **Balanced Accuracy** and **F1**, making it the strongest option for this imbalanced dataset.

### XGBoost threshold tuning (example)

- Default `0.50` threshold: **Accuracy = 0.8023**, **Recall = 0.9717**
- Tuned threshold `≈ 0.73` (from train CV to optimize F1/PR trade-off): **Accuracy = 0.8612**, **Recall = 0.7432**

---

## Feature pruning (Mutual Information)

- Original features: **5**
- Pruned features: **2** (top-k where `k = floor(n_features/2)`)
- Top features selected:
  - `absolute_magnitude`
  - `est_diameter_max`

Pruned model behavior:
- **KNN:** Accuracy slightly decreased, Balanced Accuracy ~unchanged.
- **Logistic Regression:** Performance collapsed (predicting the majority class at the tuned setting), showing sensitivity under aggressive pruning.

---

## How to run

### Option A — Google Colab (recommended)

1. Open `nasa_neos_classification.ipynb` in Colab.
2. Provide Kaggle credentials:
   - Set environment variables `KAGGLE_USERNAME` and `KAGGLE_KEY`, **or**
   - Upload `kaggle.json` and configure Kaggle.
3. Run cells top-to-bottom.

> The notebook attempts Kaggle download first. If credentials are missing, it tries a fallback download route.

### Option B — Local environment

1. Clone this repo and create a virtual environment.

2. Install dependencies:

   ```bash
   pip install -U pandas numpy scikit-learn scipy xgboost kaggle opendatasets
   ```

3. Download the dataset from Kaggle and place `neo.csv` where the notebook expects it (edit `data_dir` if needed).

4. Run the notebook:

   ```bash
   jupyter notebook nasa_neos_classification.ipynb
   ```

---

## Methods and implementation notes

- **Train/test split:** Stratified (80/20)
- **Preprocessing:** Imputation + scaling (and encoding if categorical columns exist)
- **Model selection:**
  - KNN + Logistic Regression: `GridSearchCV` with stratified 5-fold CV
  - XGBoost: `RandomizedSearchCV`, uses `scale_pos_weight` to address class imbalance
- **Threshold tuning:** Uses cross-validated probabilities on the training set to select a decision threshold that maximizes F1

---

## Repository structure

```
.
├── nasa_neos_classification.ipynb
└── README.md
```

---

## Future improvements (if extended)

- Add calibration (Platt / isotonic) and compare calibrated PR curves
- Use cost-sensitive learning or explicit recall constraints (e.g., “recall ≥ X”)
- Add interpretability (feature importance / SHAP for XGBoost)
- Add a small CLI/script to train and evaluate without a notebook

---

## Acknowledgements

- Dataset provided via Kaggle (`sameepvani/nasa-nearest-earth-objects`)
- Built with scikit-learn and XGBoost
