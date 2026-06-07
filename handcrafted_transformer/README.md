# handcrafted_transformer

A **GPT-style transformer built from scratch** using [Burn 0.21](https://burn.dev) (tracel-ai).  
Every component is written explicitly so you can follow the maths from *"Attention Is All You Need"* line by line.

---

## Requirements

- **Rust ≥ 1.85** (Burn 0.21 and its transitive dependencies require edition 2024 support)
- Install via [rustup](https://rustup.rs): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

---

## Quick start

```bash
# 1. Train on the bundled corpus (≈ 30 epochs, ~1–2 min on CPU)
cargo run --release --bin train

# 2. Generate text from the trained model
cargo run --release --bin generate
```

Training artifacts (model weights, vocabulary, config) are saved to `./artifacts/`.

---

## Project structure

```
handcrafted_transformer/
├── Cargo.toml
├── data/
│   └── corpus.txt          ← training text (swap with anything you like)
└── src/
    ├── lib.rs
    ├── tokenizer.rs         ← word-level tokenizer with special tokens
    ├── dataset.rs           ← sliding-window LM dataset + Burn Batcher
    ├── model.rs             ← THE TRANSFORMER (all components explicit)
    ├── training.rs          ← TrainStep/ValidStep, Learner wiring
    ├── generation.rs        ← greedy / sampling / top-k decoding
    └── bin/
        ├── train.rs         ← `cargo run --bin train`
        └── generate.rs      ← `cargo run --bin generate`
```

---

## Architecture — what you are building

```
token ids  [batch, seq_len]
    │
    ▼
Embedding  [vocab_size → d_model]          ← learned lookup table
    │
    + PositionalEncoding (sinusoidal)      ← fixed, not learned
    │
    ▼
┌─────────────────────────────┐
│  TransformerBlock  × n_layers│
│  ┌──────────────────────┐   │
│  │  LayerNorm           │   │   ← pre-LN variant (more stable)
│  │  MultiHeadSelfAttn   │   │   ← Q, K, V projections + scaled dot-product
│  │  + residual          │   │
│  │  LayerNorm           │   │
│  │  FeedForward (GELU)  │   │   ← 2 linear layers, inner dim = d_ff
│  │  + residual          │   │
│  └──────────────────────┘   │
└─────────────────────────────┘
    │
    ▼
LayerNorm → Linear(d_model, vocab_size)    ← language-model head
    │
    ▼
logits  [batch, seq_len, vocab_size]
```

Default hyper-parameters (tiny, trains fast):

| Parameter    | Value | Notes                          |
|-------------|-------|--------------------------------|
| `d_model`   | 128   | embedding / hidden dimension   |
| `n_heads`   | 4     | attention heads (d_k = 32)    |
| `n_layers`  | 2     | stacked transformer blocks     |
| `d_ff`      | 256   | feed-forward inner dim (2×)   |
| `seq_len`   | 32    | tokens per training window     |
| `dropout`   | 0.1   | applied in attention + FFN     |
| `epochs`    | 30    | Adam, lr = 1e-3                |

---

## Key concepts implemented

### Scaled dot-product attention
```
Attention(Q, K, V) = softmax( Q @ K^T / √d_k ) @ V
```
Implemented in `MultiHeadSelfAttention::forward` in `model.rs`.

### Causal (autoregressive) mask
An upper-triangular mask of −1×10⁹ added to attention scores before softmax,
preventing position *i* from attending to any position *j > i*.

### Positional encoding
```
PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```
Pre-computed once in `build_sinusoidal_encoding`, stored as a non-parameter tensor.

### Residual connections + Pre-LayerNorm
```
x = x + Attention(LayerNorm(x))
x = x + FFN(LayerNorm(x))
```

### Cross-entropy loss
```
loss = -mean( log_softmax(logits)[target_token] )
```

### Text generation
Three strategies in `generation.rs`:
- **Greedy** — argmax at each step
- **Temperature sampling** — softmax(logits / T), then sample
- **Top-k sampling** — restrict to k highest logits, then temperature-sample

---

## Swapping in your own corpus

Replace `data/corpus.txt` with any plain-text file.  
The tokenizer splits on whitespace and lowercases everything — no external tokenizer library needed.

For a larger corpus (books, Wikipedia extracts) you'll want to increase:
```toml
d_model   = 256
n_heads   = 8
n_layers  = 4
d_ff      = 1024
seq_len   = 128
num_epochs = 50
```

---

## Understanding the loss curve

A freshly initialised model has loss ≈ `ln(vocab_size)` (random chance).  
After training on the bundled corpus you should see it drop to ~1.5–2.5.  
The vocabulary is small (~250 words) so the model can memorise patterns quickly.

---

## Burn-specific notes

- **Backend**: `NdArray<f32>` (pure CPU, no CUDA required)  
- **AutodiffBackend**: `Autodiff<NdArray<f32>>` wraps the base backend with reverse-mode AD  
- **Learner**: Burn's built-in training loop with terminal UI, checkpointing, and metrics  
- **Module derive macro**: `#[derive(Module)]` automatically handles parameter registration,  
  device movement, and train/eval mode switching  
- **Config derive macro**: `#[derive(Config)]` makes hyper-parameters serialisable to JSON  
