# Step 8: Building Adam (The Optimizer)

An **optimizer** is just an object that owns the weight-update rule. We build two: `Sgd` and `Adam`.

## 💡 The Analogy
- **SGD** (Step 7, repackaged): a hiker taking equal-sized steps downhill.
- **Adam**: a smarter hiker with two upgrades —
  - **momentum** (`m`): keep some speed from previous steps, to roll through small bumps.
  - **adaptive steps** (`v`): big gradients get *smaller* steps, tiny gradients get *bigger* steps,
    so every weight moves at a sensible pace.

## 📐 The Adam update (per weight)
```
m = β1·m + (1-β1)·g            # momentum: smoothed gradient
v = β2·v + (1-β2)·g²           # variance: smoothed squared gradient
m̂ = m / (1 - β1^t)            # bias-correct (so early steps aren't tiny)
v̂ = v / (1 - β2^t)
weight -= lr · m̂ / (√v̂ + ε)   # adaptive step
```
Defaults `β1=0.9, β2=0.999, ε=1e-8` match Burn and PyTorch.

## 🧪 The demo
Trains two **identical** models (same seed) — one with `Sgd`, one with `Adam` — and prints both loss
curves side by side so you can see Adam's faster early dive.

## 🔗 Burn bridge
| Here | In `handcrafted_transformer` |
|------|------------------------------|
| `Adam::new(lr)` | `AdamConfig::new()` |
| `adam.step(&params)` | what `Learner` calls internally each batch |

## 🚀 How to Run
```bash
cargo run
```
Prints an SGD-vs-Adam loss comparison over 30 steps.
