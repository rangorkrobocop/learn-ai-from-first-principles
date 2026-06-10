# Step 6: The Scorecard (Loss)

We need a single number that says how wrong the model is right now. That's the **loss**.

## 💡 The Analogy
The **loss** squeezes every mistake into one number. We use **Mean Squared Error**:
$$\text{MSE} = \frac{1}{N}\sum_{i=1}^{N}(\text{guess}_i - \text{answer}_i)^2$$
- Guess matches answer → that term is `0` (no penalty).
- Guess far off → squaring makes the penalty grow fast.

A perfect model scores `0`. Training is just: make this number small.

## 🔑 The key trick
The loss is itself a `Value`. So `loss.backward()` flows slopes back to **every weight in the
model** in one call — the same move the transformer makes with its cross-entropy loss
(`out.loss.backward()` in `training.rs`).

## 🧪 The demo
Andrej Karpathy's tiny 4-example dataset (each example wants `+1` or `-1`). We forward all four,
compute the MSE, then call `backward()` and confirm every parameter now has a non-zero gradient —
proof there's a downhill direction to follow.

## 🔗 Burn bridge
| Here | In `handcrafted_transformer` |
|------|------------------------------|
| `mse(preds, targets)` | `CrossEntropyLossConfig::new().init(..).forward(..)` |
| `loss.backward()` | `out.loss.backward()` |

## 🚀 How to Run
```bash
cargo run
```
Prints the untrained predictions, the loss, and confirms `backward()` graded all parameters.
