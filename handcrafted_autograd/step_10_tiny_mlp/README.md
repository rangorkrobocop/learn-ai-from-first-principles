# Step 10: Train a Real Net (The Tiny MLP) — Graduation 🎓

The finale: a complete neural network, trained by an engine you wrote from nothing.

## 💡 The Analogy
We make a 2D dataset — dots **inside** a circle (class `+1`) vs **outside** (class `-1`) — and train
an MLP to tell them apart. A circle isn't a straight line, so the network has to **bend** its
decision boundary. At the end we draw that boundary in ASCII so you can *see* what it learned.

## 🧩 What's assembled here
Every block is something you built in Steps 1–9, stacked together:
- **Value + backward** (Steps 1–3) — the autograd engine
- **Module / Linear / Mlp** (Steps 4–5) — a `2 → [16, 16, 1]` network
- **MSE loss** (Step 6) — the scorecard
- **Adam** (Step 8) — the optimizer
- **the training loop** (Steps 7, 9) — forward → loss → zero → backward → step

## 🗺️ The output
A loss/accuracy log, then an ASCII map where `#` is the region the model calls "inside," `·` is
"outside," and the `o` ring marks the *true* circle. A trained model's `#` blob hugs the `o` ring.

## 🔗 Burn bridge
Swap `Value` for `Tensor`, this 2-input net for a 12-million-parameter GPT, and "inside the circle"
for "the next word," and you have `handcrafted_transformer`. **Same engine, bigger numbers.**

## 🚀 How to Run
```bash
cargo run --release      # release is ~10× faster for this one
```

> ⏱️ The scalar engine builds a fresh graph of ~30k nodes each epoch, so use `--release`. This is
> exactly *why* real frameworks like Burn use batched **tensors** instead of scalars — same math,
> vastly faster. You've now felt the reason firsthand.
