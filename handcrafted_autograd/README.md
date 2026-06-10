# 🧠 Handcrafted Autograd: The Engine Underneath Your Transformer

Welcome back, Future AI Engineer!

In **`handcrafted_transformer`** you built a real GPT — attention, masking, the whole factory.
But you let a library called **Burn** do three magic tricks for you:

1. `loss.backward()` — figure out how to nudge *every* weight (backpropagation)
2. `AdamConfig` — the optimizer that actually does the nudging
3. `Learner` — the training loop that repeats it all

This project rips the lid off those three boxes. We build a tiny **automatic differentiation engine**
from scratch in Rust — the same idea as Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) —
and use it to train a real (tiny) neural network. After this, **nothing in the transformer is magic anymore.**

> ⚠️ This is a **conceptual prerequisite**, not a code dependency. The transformer is built on Burn's
> *tensors* (whole arrays at once); our engine works on single *scalars* (one number at a time) so you can
> watch every gradient by hand. It teaches what Burn does — it does not get `use`d by the transformer.

---

## 🗺️ The Map of Our Journey

```
step_01_value/             ➔ A number that remembers it has a slope (grad)
       │
step_02_operations/        ➔ Teach +, ×, tanh to remember their parents (build a graph)
       │
step_03_backprop/          ➔ The chain rule: pour gradients backwards through the graph
       │
step_04_neuron/            ➔ One neuron = inputs · weights + bias, then squash
       │
step_05_layer_mlp/         ➔ Stack neurons into Layers into a brain (an MLP)
       │
step_06_loss/              ➔ A scorecard (MSE): how wrong are we right now?
       │
step_07_gradient_descent/  ➔ Nudge every weight downhill — by hand
       │
step_08_optimizer/         ➔ Build Adam yourself (momentum + adaptive steps)
       │
step_09_train_loop/        ➔ The tutor: forward → loss → backward → step, repeat
       │
step_10_tiny_mlp/          ➔ Train an MLP to classify a 2D dataset & draw the boundary!
```

Each step is its own standalone crate. `cd` into any folder and run `cargo run`.

---

## 🦀 A note on the word "interface"

In Rust, the word for an **interface** — a shared set of methods many types promise to provide — is a
**`trait`**. That's exactly the mechanism Burn uses: its `#[derive(Module)]` implements a `Module` trait.

To keep this engine familiar, we deliberately mirror Burn's shapes:

| Burn (in `handcrafted_transformer`)        | Handcrafted autograd (here)              |
|--------------------------------------------|------------------------------------------|
| `trait Module` + `#[derive(Module)]`       | `trait Module { forward, parameters }`   |
| `LinearConfig::new(a, b).init(&device)`    | `LinearConfig::new(a, b).init(&mut rng)` |
| `Linear`, `Embedding`, `LayerNorm`         | `Linear`, `Mlp`                          |
| `model.forward(x)`                         | `model.forward(&x)`                      |
| `out.loss.backward()`                      | `loss.backward()`                        |
| `AdamConfig::new().init()` + `Learner`     | `Adam::new(lr)` + hand-written loop      |
| `Tensor<B, D>` (whole arrays)              | `Value` (single scalars)                 |

The only essential difference is **scalar vs. tensor**. Burn batches thousands of numbers per op for speed;
we use one number per op for clarity. The *math* — and the `trait`s around it — is the same.

---

## 🚀 Quick start

```bash
# run any step
cargo run --manifest-path step_03_backprop/Cargo.toml

# or, with the `just` task runner (see justfile):
just run 3       # run step 3
just run 10      # the grand finale
just build       # compile every step
```

---

## 🏆 What you'll understand at the end

The exact line `out.loss.backward()` in `handcrafted_transformer/.../training.rs` — what it computes,
why it works, and why Adam updates weights the way it does. You'll have written all of it yourself.

Happy back-propagating! 🧠
