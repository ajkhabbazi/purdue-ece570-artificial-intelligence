# Transformers (Decoder-Only) Language Model — Tiny Shakespeare

A single-notebook assignment implementing a small **decoder-only Transformer** language model trained on **Tiny Shakespeare**. The notebook covers **tokenization (char/word/BPE)**, **positional encoding**, **causal multi-head self-attention**, **training**, and **text generation** with **temperature / top-k / top-p** sampling.

## Project Files

- `transformers.ipynb` — main notebook (implementation + experiments)

## Features Implemented

- Tiny Shakespeare data download/loading
- Tokenizers:
  - Character-level tokenizer
  - Word-level tokenizer (with `<UNK>`)
  - BPE tokenizer (via Hugging Face `tokenizers`)
- Model:
  - Sinusoidal positional encodings
  - Causal (masked) multi-head self-attention
  - Feed-forward MLP blocks
  - Stacked decoder blocks → logits over vocabulary
- Training:
  - Next-token prediction (cross-entropy)
  - Optimizer: AdamW
- Generation:
  - Temperature sampling
  - Top-k sampling
  - Top-p (nucleus) sampling

## Requirements

- Python 3.9+ (recommended)
- Packages:
  - `torch`
  - `numpy`
  - `requests`
  - `tokenizers`

Install:

```bash
pip install -U torch numpy requests tokenizers
```

> GPU is optional. The notebook will use `cuda` if available.

## How to Run

1. Open `transformers.ipynb` in Jupyter / VS Code / Colab.
2. Run all cells.

The notebook prints training logs and produces sample generations for multiple decoding configurations.

## Default Hyperparameters (as set in the notebook)

- `context_length = 128`
- `d_model = 384`
- `n_layers = 4`
- `n_heads = 8`
- `dropout_rate = 0.1`
- `batch_size = 32`
- `num_epochs = 2`
- `learning_rate = 3e-4`
- BPE vocab size: `3000`

## Generation Settings Used

- Prompt: `O Romeo, Romeo, wherefore art thou Romeo?`
- `max_new_tokens = 200`
- Temperature: `T = 0.2`, `T = 1.0`
- Top-k: `k = 5`, `k = 50`
- Top-p: `p = 0.80`, `p = 0.95`

## Notes

This is an educational implementation. For stronger generations, increase training time/epochs and/or model capacity (layers, `d_model`, context length).

## Acknowledgements

- Tiny Shakespeare dataset (via Andrej Karpathy `char-rnn`)
- Hugging Face `tokenizers`
- PyTorch
