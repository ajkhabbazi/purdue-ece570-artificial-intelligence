# GRPO Fine-Tuning on a Toy Addition Task

A minimal, self-contained notebook that fine-tunes a small causal language model (**`distilgpt2`**) with **Group Relative Policy Optimization (GRPO)** on a toy task: **single-step addition of two one-digit integers**.

The goal is to demonstrate the full RL fine-tuning loop end-to-end (data → reward → GRPO update → evaluation) in a way that’s easy to run and inspect.

## Project contents

- `grpo_fine_tuning.ipynb` — the full assignment notebook (task definition, dataset generation, reward, GRPO step, training loop, evaluation, and analysis).

## Task

- **Prompt template:** `Add the numbers: {a} + {b} =`
- **Desired output:** the correct sum as a plain decimal string (e.g., `12`)

## Method: GRPO (high level)

For each prompt, the current policy samples a **group** of completions (size **G**) and receives rewards. Rewards are normalized *within the group* to form relative advantages, and the policy is updated to increase the likelihood of higher-advantage completions.

This notebook also includes an **optional KL penalty** against a frozen reference model (disabled by default) to keep the fine-tuned policy close to the starting model.

## Reward

The implemented reward is intentionally simple:
- Parse the **first integer** in the model’s generated text.
- Reward = `1.0` if it matches the correct sum, else `0.0`.

> Note: This makes the task easy to optimize, but it can also allow “messy” outputs that start with the correct number (see the Analysis section in the notebook).

## Results (from the notebook run)

- **Baseline (frozen reference) mean eval reward:** ~0.02  
- **After GRPO fine-tuning mean eval reward:** ~0.25

See **Cell 8: Analysis** in the notebook for qualitative behavior and discussion of reward dynamics.

## Getting started

### Requirements

- Python 3.10+ recommended
- PyTorch
- Hugging Face `transformers`

Install (example):

```bash
pip install torch transformers
```

> If you are on Apple Silicon or CUDA, install the appropriate PyTorch build for your system.

### Run

Open and run the notebook:

```bash
jupyter notebook grpo_fine_tuning.ipynb
```

The notebook is organized into numbered cells:
1. Task definition and plan  
2. Setup and model loading  
3. Dataset generation  
4. Reward implementation  
5. GRPO step  
6. Training loop  
7. Evaluation and examples  
8. Analysis  

## Key hyperparameters (as used in the notebook)

- `MODEL_NAME`: `distilgpt2`
- `N_TRAIN`: 1000, `N_EVAL`: 200
- `TRAIN_STEPS`: 200
- `BATCH_SIZE`: 16
- `LEARNING_RATE`: 1e-5
- `GROUP_SIZE (G)`: 4
- `MAX_NEW_TOKENS`: 5
- `TEMPERATURE`: 1.0
- `TOP_K`: 50
- `KL_COEF`: 0.0 (set > 0.0 to enable KL regularization)

## Reproducibility

The notebook sets a fixed random seed and runs on CPU or GPU depending on availability. Results can vary slightly across hardware and library versions due to sampling and floating-point differences.

## Known limitations / next steps

- **Reward shaping:** The current reward checks only the first integer, which can incentivize outputs like `"7 = 8 = 9"`. A stricter reward (exact match, regex constraints, or formatting penalties) would better align behavior with the intended output.
- **Better evaluation:** Add exact-match accuracy, calibration checks, and error breakdowns by operand pairs.
- **Stability:** Try enabling a small KL coefficient and/or experimenting with larger group sizes.

## Acknowledgements

- PyTorch
- Hugging Face Transformers

## License

This repository is intended for coursework/educational use. If you plan to reuse or redistribute, add an explicit open-source license file (e.g., MIT, Apache-2.0) that matches your intended use.
