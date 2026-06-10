# Step 4: One Brain Cell (The Neuron)

We package the loose values from Step 3 into a reusable `Neuron` with a `forward()` method.

## 💡 The Analogy
A **neuron** takes several inputs, multiplies each by a **weight**, adds a **bias**, then squashes
the total with `tanh`:
$$\text{output} = \tanh(w_1 x_1 + w_2 x_2 + \dots + b)$$
It's a weighted vote followed by a "how excited am I?" knob.

## 🔩 The parts
- `w` — one **weight** per input (how much that input matters).
- `b` — one **bias** (a baseline nudge).
- `forward(x)` — runs the vote and the squash.
- `parameters()` — returns every tunable knob, exactly what an optimizer will nudge later.

## 🔗 Burn bridge
This `forward()` / `parameters()` pair is the shape Burn gives **every** layer in the transformer
(`self.w_q.forward(x)` in `model.rs`). We're matching that interface deliberately.

## 🚀 How to Run
```bash
cargo run
```
Builds one neuron, runs a forward pass, then `backward()`, and prints the gradient on each weight and
the bias. (The engine code at the top of `main.rs` is identical to Step 3.)
