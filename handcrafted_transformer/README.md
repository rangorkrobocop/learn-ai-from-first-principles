# 🚀 Handcrafted Transformer: 10-Step Journey to Building Your Own AI

Welcome, Future AI Engineer!

Have you ever wondered how ChatGPT reads your questions and writes back like a human? Inside ChatGPT
is a special math engine called a **Transformer**. In this journey we build a mini-Transformer
**from scratch in Rust** — a GPT-style, decoder-only model where every component is written out
explicitly so you can follow the maths from *"Attention Is All You Need"* line by line.

We start with the absolute basics (turning words into numbers) and finish with a model that writes
its own stories. Built on [Burn 0.21](https://burn.dev) (tracel-ai).

> 🧠 Curious what `loss.backward()` and `Adam` are *actually* doing under the hood? The sibling
> project **`../handcrafted_autograd`** builds that engine from scratch — it's the conceptual
> prerequisite to this one.

---

## 🗺️ The Map of Our Journey

```
step_01_tokenizer/             ➔ Turn words into secret code numbers
       │
step_02_dataset/               ➔ Make practice flashcards from a book
       │
step_03_embedding/             ➔ Place words on a 2D map (GPS coordinates)
       │
step_04_positional_encoding/   ➔ Add page numbers so order matters
       │
step_05_attention/             ➔ Shine a flashlight on related words
       │
step_06_causal_masking/        ➔ Cover the future (no cheating!)
       │
step_07_multi_head_attention/  ➔ Form a team of specialized detectives
       │
step_08_feed_forward/          ➔ Add a brainy decision-making layer
       │
step_09_transformer_block/     ➔ Build a factory assembly line
       │
step_10_complete_gpt/          ➔ Practice with a tutor & write stories!
```

**Each step is its own independent crate.** `cd` into any step and run `cargo run` to see it in
action — or use the `just` task runner (`just run 5`, `just build`).

---

## Requirements

- **Rust ≥ 1.85** (Burn 0.21 and its transitive dependencies require edition 2024 support)
- Install via [rustup](https://rustup.rs): `curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh`

---

## 🕵️‍♂️ Step 1: The Secret Decoder Ring (The Tokenizer)

### 💡 The Analogy
Computers are giant calculators — they don't understand letters or words, only numbers! A
**Tokenizer** is like a **Secret Decoder Ring**. It takes a sentence, cleans up the punctuation,
splits it into words, and assigns a unique number code to every word it has ever seen.

```bash
cd step_01_tokenizer && cargo run
```

---

## 🗂️ Step 2: Training Flashcards (The Dataset)

### 💡 The Analogy
How do you study for a spelling test? You use **flashcards**! On the front, a clue; on the back, the
answer. To train our AI, we make thousands of sliding-window flashcards from a book. Every target
word is just the input sequence shifted one word into the future!

```bash
cd step_02_dataset && cargo run
```

---

## 📍 Step 3: Words on a GPS Map (Embeddings)

### 💡 The Analogy
If you just give words random numbers, the computer doesn't know that *"apple"* and *"banana"* are
both fruits. An **Embedding** turns each number code into **GPS coordinates on a map** where similar
words are placed close together! We use the Pythagorean theorem to measure distances.

```bash
cd step_03_embedding && cargo run
```

---

## 📖 Step 4: Adding Page Numbers (Positional Encoding)

### 💡 The Analogy
Without order, *"the dog bit the boy"* and *"the boy bit the dog"* look identical to the model. We
add **Positional Encodings** — smooth mathematical waves (sines and cosines) that create a unique
fingerprint for every position in a sentence.

```bash
cd step_04_positional_encoding && cargo run
```

---

## 🔦 Step 5: Focusing Your Flashlight (Single-Head Attention)

### 💡 The Analogy
When you read **"bank"** in *"I sat by the river bank"*, how do you know it's not a money bank? You
look at the word **"river"**! **Attention** is like shining a flashlight. We use dot products to find
how well words match, and softmax to turn scores into percentages.

```bash
cd step_05_attention && cargo run
```

---

## 🚫 Step 6: No Cheating! (Causal Masking)

### 💡 The Analogy
If you're taking a next-word prediction test, you can't look at the answer page! We use a **Causal
Mask** to block the AI from looking into the future. By adding negative infinity (−1,000,000,000) to
future attention scores, their probability becomes exactly 0%.

```bash
cd step_06_causal_masking && cargo run
```

---

## 👥 Step 7: A Team of Detectives (Multi-Head Attention)

### 💡 The Analogy
If you only have one flashlight, you can only focus on one thing. So we build a **team of detectives**
(called **heads**). Each detective works independently in their own room, then they gather to combine
their notes into one master report.

```bash
cd step_07_multi_head_attention && cargo run
```

---

## 🧠 Step 8: The Brain's Decision Filter (Feed-Forward & GELU)

### 💡 The Analogy
After our detectives gather their clues, the AI needs to *think* and make a decision. The
**Feed-Forward Network** with **GELU** activation acts like a dimmer switch: if a number is negative
(unimportant), it dims it to zero; if positive, it lets it pass. This is where the model gains the
power to learn complex rules.

```bash
cd step_08_feed_forward && cargo run
```

---

## 🏗️ Step 9: The Factory Assembly Line (The Transformer Block)

### 💡 The Analogy
A **Transformer Block** is a single station in our factory:
1. Normalize (LayerNorm)
2. Inspect context (Attention) + keep the blueprint (Residual Connection)
3. Normalize again
4. Think and filter (Feed-Forward) + keep the blueprint again

By linking multiple blocks together, the model gets smarter at each station!

```bash
cd step_09_transformer_block && cargo run
```

---

## 🎓 Step 10: Training & Storytime (The Complete GPT Model)

### 💡 The Analogy
Now it's school time! We assemble our complete model:
* **The Student**: the Transformer.
* **The Tutor**: the training loop.
* **The Scorecard**: the Loss Function (Cross-Entropy).
* **The Coach**: the Optimizer (Adam).

We train on a tiny text corpus, watch the loss score drop, and then ask the model to finish prompts!

```bash
cd step_10_complete_gpt
cargo run --release --bin train      # Train the model (≈ 30 epochs, ~1–2 min on CPU)
cargo run --release --bin generate   # Generate stories!
```

Training artifacts (model weights, vocabulary, config) are saved to `./artifacts/`.

---

## 🔬 Under the Hood — the complete GPT (step 10)

Once you understand the 10 steps, here's how the final model in `step_10_complete_gpt` fits together.

### Project structure
```
step_10_complete_gpt/
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

### Architecture — what you are building
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

### Default hyper-parameters (tiny, trains fast)

| Parameter    | Value | Notes                          |
|-------------|-------|--------------------------------|
| `d_model`   | 128   | embedding / hidden dimension   |
| `n_heads`   | 4     | attention heads (d_k = 32)     |
| `n_layers`  | 2     | stacked transformer blocks     |
| `d_ff`      | 256   | feed-forward inner dim (2×)    |
| `seq_len`   | 32    | tokens per training window     |
| `dropout`   | 0.1   | applied in attention + FFN     |
| `epochs`    | 30    | Adam, lr = 1e-3                |

---

## 📐 Key concepts implemented

### Scaled dot-product attention
```
Attention(Q, K, V) = softmax( Q @ K^T / √d_k ) @ V
```
Implemented in `MultiHeadSelfAttention::forward` in `model.rs`.

### Causal (autoregressive) mask
An upper-triangular mask of −1×10⁹ added to attention scores before softmax, preventing position
*i* from attending to any position *j > i*.

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

## 🔁 Swapping in your own corpus

Replace `data/corpus.txt` with any plain-text file. The tokenizer splits on whitespace and
lowercases everything — no external tokenizer library needed.

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

## 📉 Understanding the loss curve

A freshly initialised model has loss ≈ `ln(vocab_size)` (random chance). After training on the
bundled corpus you should see it drop to ~1.5–2.5. The vocabulary is small (~250 words), so the model
can memorise patterns quickly.

---

## 🦀 Burn-specific notes

- **Backend**: `NdArray<f32>` (pure CPU, no CUDA required)
- **AutodiffBackend**: `Autodiff<NdArray<f32>>` wraps the base backend with reverse-mode AD
- **Learner**: Burn's built-in training loop with terminal UI, checkpointing, and metrics
- **Module derive macro**: `#[derive(Module)]` automatically handles parameter registration, device
  movement, and train/eval mode switching
- **Config derive macro**: `#[derive(Config)]` makes hyper-parameters serialisable to JSON

---

## 🏆 What Next?
Once you understand these 10 steps, you understand the core math of modern AI! Each step folder has
its own `README.md` with deeper explanations and basic math breakdowns.

Want to demystify the last piece of magic — how gradients and Adam actually work? Head to
**`../handcrafted_autograd`** and build the autodiff engine underneath all of this from scratch.

Happy coding! 🚀
