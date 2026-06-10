# Step 7: Walking Downhill By Hand (Gradient Descent)

Now we actually make the model better — by stepping downhill on the loss.

## 💡 The Analogy
The loss is a hill; we want the valley. `backward()` tells us which way is **uphill** for each
weight (its `grad`). So we step the **opposite** way:
$$\text{weight} \leftarrow \text{weight} - \text{learning\_rate} \times \text{grad}$$
Small steps, repeated, walk the loss down.

## ⚠️ The #1 autograd bug: forgetting `zero_grad`
Gradients **accumulate** (our engine uses `+=`). If you don't reset them before each `backward()`,
yesterday's slopes pollute today's and training goes haywire. The loop must be:

```
forward → zero_grad → backward → step
```

## 🧪 The demo
Trains the `3 → [4,4,1]` MLP on Karpathy's 4-example dataset for 30 hand-written steps. You'll watch
the loss fall and the final predictions snap close to the desired `±1`.

## 🔗 Burn bridge
This five-line loop is exactly what Burn's `Learner` automates for the transformer. The
`weight -= lr * grad` rule is plain **SGD** — the simplest optimizer. Step 8 upgrades it to Adam.

## 🚀 How to Run
```bash
cargo run
```
Prints the loss every few steps and the final predictions.
