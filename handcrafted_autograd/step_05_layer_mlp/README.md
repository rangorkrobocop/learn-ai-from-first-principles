# Step 5: Stacking Cells Into a Brain (Layers & the MLP)

One neuron is weak. We stack many into layers, and layers into a network.

## 💡 The Analogy
- A **Layer** (`Linear`) is a row of neurons that all see the same inputs.
- An **MLP** (Multi-Layer Perceptron) stacks layers, so one layer's output feeds the next —
  a brain built from brain cells.

Hidden layers use a `tanh` squash; the final layer stays linear so it can output any raw score.

## 🦀 The `Module` trait (a.k.a. "the interface")
In Rust, an interface is a **`trait`**. We define:
```rust
pub trait Module {
    fn forward(&self, x: &[Value]) -> Vec<Value>;
    fn parameters(&self) -> Vec<Value>;
}
```
Both `Linear` and `Mlp` implement it. This is the exact idea behind Burn's `#[derive(Module)]` in
the transformer — every learnable block promises a `forward` and a list of `parameters`.

We also copy Burn's **builder** style: `LinearConfig::new(d_in, d_out).with_nonlin(true).init(&mut rng)`,
mirroring `LinearConfig::new(a, b).init(&device)` in `model.rs`.

## 🔗 Burn bridge
| Here | In `handcrafted_transformer` |
|------|------------------------------|
| `trait Module { forward, parameters }` | `#[derive(Module)]` |
| `LinearConfig::new(a,b).init(&mut rng)` | `LinearConfig::new(a,b).init(&device)` |
| `Mlp` | `HandcraftedTransformer` |

## 🚀 How to Run
```bash
cargo run
```
Builds a `3 → [4,4,1]` MLP with a fixed random seed, runs a forward pass, counts parameters, and
shows that `backward()` still flows through the whole stack.
