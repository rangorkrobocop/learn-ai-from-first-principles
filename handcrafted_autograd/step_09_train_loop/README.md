# Step 9: The Tutor (The Training Loop)

Every piece we built has one natural rhythm. We wrap it in a `Learner`.

## 💡 The Analogy
The tutor drills the student in a fixed cycle, every epoch:
```
forward → loss → zero_grad → backward → optimizer step
```
That's the entire job of a training loop. We package it in a `Learner` that owns a model and an
optimizer and exposes `.fit(xs, ys, epochs)` and `.predict(x)`.

## 🧪 The demo
Builds a `3 → [4,4,1]` MLP, hands it to a `Learner` with an `Adam` optimizer, and calls
`learner.fit(..)` for 40 epochs. The loss drops and the predictions converge to the target `±1`.

## 🔗 Burn bridge
This is a direct miniature of the transformer's training entry point:
| Here | In `handcrafted_transformer/.../training.rs` |
|------|----------------------------------------------|
| `Learner::new(model, optim)` | `Learner::new(model, optim.init(), lr)` |
| `learner.fit(xs, ys, epochs)` | `SupervisedTraining::new(..).num_epochs(n).launch(..)` |
| `train_step` body | `TrainStep::step` → `out.loss.backward()` |

After this step, re-read `out.loss.backward()` in the transformer — there's nothing left to mystify
you.

## 🚀 How to Run
```bash
cargo run
```
Prints the loss every 5 epochs and the final predictions.
