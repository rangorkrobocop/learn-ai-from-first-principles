# Step 1: A Number With a Slope (The `Value`)

We start the whole engine with one tiny idea: a number that remembers its **slope**.

## 💡 The Analogy
A plain number like `3.0` is forgetful — it only knows its size. Our `Value` has a second slot,
`grad`, which stores its **slope**: *"if I wiggle this number up a little, how much does the final
score move?"*

Every `Value` starts with `grad = 0.0`. We haven't asked any questions yet. Step 3 will fill these
slopes in automatically with backpropagation.

## 🧠 Why two slots?
- `data` → what the model **outputs**.
- `grad` → how the model should **improve**.

Training is nothing more than: read every `grad`, then nudge every `data` a little.

## 🦀 Rust note
We wrap the number in `Rc<RefCell<..>>` so that later, many parts of one math expression can
**share and update** the same number. Read `Value` as just "a smart number" for now.

## 🔗 Burn bridge
In `handcrafted_transformer`, every weight is a `Tensor` that also secretly carries a gradient.
We're building the single-number version so you can *see* the gradient with your own eyes.

## 🚀 How to Run
```bash
cargo run
```
This creates a few `Value`s, prints their `data` and `grad`, and hand-sets one slope to show what
the slot is for.
