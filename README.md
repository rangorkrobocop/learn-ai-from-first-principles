# 🧠 Learn AI From First Principles

**Build modern AI from scratch — every layer, no black boxes.**

Most AI tutorials hand you a black box: `import torch`, call `.fit()`, and trust that the magic
works. This repo does the opposite. We rebuild a GPT-style language model **and the autodiff engine
underneath it** from nothing — in Rust — so that by the end there is no line you can't explain.

The goal is exactly what the name says: understand AI from the ground up, starting with arithmetic
and ending with a transformer that writes its own text. If you've ever wanted to *truly* know what
`loss.backward()`, "attention," and "Adam" mean — not just how to call them — this is for you.

---

## 🪜 The two projects (a deliberate ladder)

The repo is two standalone, bite-sized courses. Each is a series of numbered steps you can `cd` into
and run. Together they cover the whole stack, from the chain rule to a working GPT.

```
learn-ai-from-first-principles/
│
├── handcrafted_autograd/      ▸ PART 1 — the engine
│      Build a reverse-mode automatic differentiation engine from scratch
│      (the idea behind Karpathy's micrograd, in Rust). Scalars → backprop →
│      neurons → an MLP → Adam → training loop. This is what every deep-
│      learning framework does for you, written by hand.
│
└── handcrafted_transformer/   ▸ PART 2 — the model
       Build a GPT-style transformer component by component on the Burn
       framework. Tokenizer → embeddings → attention → masking → multi-head →
       feed-forward → the full block → a trained, text-generating model.
```

**Why two projects?** Part 2 (the transformer) uses a framework (Burn) that computes gradients for
you. Part 1 (the autograd engine) is where you *build that gradient machine yourself*. Do both and
the entire pipeline — from a single derivative to a million-parameter model — is demystified.

---

## 🗺️ Recommended path

You can start with either, but for a true first-principles arc:

1. **`handcrafted_autograd`** first — learn how learning itself works (backprop + optimizers) on
   tiny, fully-visible scalar examples.
2. **`handcrafted_transformer`** next — apply those same ideas at scale to build a real GPT.

Prefer to see the exciting payoff first? Start with the transformer, then come back to the autograd
engine to lift the lid on `loss.backward()`. Each project's own `README.md` is a complete guided tour
with analogies and the math spelled out, step by step.

| Project | What you build | Depends on |
|---------|----------------|-----------|
| [`handcrafted_autograd`](handcrafted_autograd/README.md) | A scalar autodiff engine + a trained MLP, from zero | nothing but `rand` |
| [`handcrafted_transformer`](handcrafted_transformer/README.md) | A decoder-only GPT, trained to generate text | [Burn 0.21](https://burn.dev) |

---

## 🚀 Quick start

**Requirements:** Rust ≥ 1.85 — install via [rustup](https://rustup.rs).

```bash
git clone https://github.com/rangorkrobocop/learn-ai-from-first-principles.git
cd learn-ai-from-first-principles

# Part 1 — see backpropagation work on Karpathy's classic neuron
cd handcrafted_autograd
cargo run --manifest-path step_03_backprop/Cargo.toml
cargo run --release --manifest-path step_10_tiny_mlp/Cargo.toml   # train an MLP, draw the result

# Part 2 — train a tiny GPT and generate text
cd ../handcrafted_transformer/step_10_complete_gpt
cargo run --release --bin train
cargo run --release --bin generate
```

Each project also ships a [`just`](https://github.com/casey/just) task runner — `just run 3`,
`just build`, etc. Run `just` in either folder to list recipes.

---

## 🎓 What you'll understand by the end

- **How a neural network learns** — the computation graph, the chain rule, and reverse-mode
  autodiff (`backward()`), built by hand.
- **What optimizers actually do** — SGD and Adam (momentum + adaptive steps), implemented from
  scratch.
- **The transformer architecture, in full** — token embeddings, sinusoidal positional encoding,
  scaled dot-product attention, causal masking, multi-head attention, feed-forward layers with
  GELU, residual connections, and pre-LayerNorm.
- **The training and generation loop** — cross-entropy loss, batching, and greedy / temperature /
  top-k decoding.
- **Why frameworks exist** — after computing gradients one scalar at a time, you'll *feel* why real
  systems batch everything into tensors.

---

## 🛠️ Why Rust?

No hidden Python magic, no autograd you didn't write. Rust forces every data flow to be explicit,
which is perfect for learning: when the gradient moves, you can see exactly where it goes. It's also
fast enough to train these small models on a laptop CPU with no GPU required.

---

## 📚 Inspiration & credits

- Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) and the *makemore* / *nanoGPT*
  lineage — the spirit of "build it small, build it yourself."
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) — the transformer.
- [Burn](https://burn.dev) (tracel-ai) — the Rust deep-learning framework powering Part 2.

---

## 🤝 Contributing & license

This is a learning repository — issues, corrections, and clearer explanations are very welcome.
Each project folder is self-contained, so improvements can target a single step.

Happy first-principles building! 🧠🚀
