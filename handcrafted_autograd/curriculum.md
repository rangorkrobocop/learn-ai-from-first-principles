# 🧠 Handcrafted Autograd: 10 Steps to Backpropagation from Scratch

This is the **prequel** to `handcrafted_transformer`. There, Burn computed gradients for you
(`out.loss.backward()`). Here, you build that machine yourself, scalar by scalar, the way
Andrej Karpathy's [micrograd](https://github.com/karpathy/micrograd) does — but in Rust, and
with names that echo Burn so the bridge is obvious.

The whole engine is just two ideas:
1. **Every number remembers how it was made** (a graph of operations).
2. **The chain rule pours slopes backwards through that graph** (backpropagation).

Everything else — neurons, layers, loss, Adam, the training loop — is built on top.

---

## 🔢 Step 1: A Number With a Slope (The `Value`)

### 💡 The Analogy
A normal number like `3.0` is forgetful. Our `Value` is a number that *also* carries a second
slot called `grad` — its **slope**: "if I wiggle this number up a little, how much does the final
score change?" Right now the slope is `0`; later, backprop fills it in.

```bash
cd step_01_value && cargo run
```

---

## ➕ Step 2: Remembering Your Parents (Operations & the Graph)

### 💡 The Analogy
When you compute `e = a * b`, the result `e` should remember that it came from `a` **times** `b`.
We teach `+` and `*` to record their **parents** and the **operation**. String enough of these
together and you've drawn a family tree — the **computation graph**.

```bash
cd step_02_operations && cargo run
```

---

## 🌊 Step 3: Pouring Gradients Backwards (Backpropagation)

### 💡 The Analogy
Water flows downhill; gradients flow *backwards*. We seed the final answer with a slope of `1.0`,
then walk the graph in reverse, and at every node the **chain rule** multiplies the slopes together.

### 📐 Basic Mathematics: The Chain Rule
If `c` depends on `b`, and `b` depends on `a`, then:
$$\frac{dc}{da} = \frac{dc}{db} \times \frac{db}{da}$$
We rebuild Karpathy's famous tanh-neuron and check the gradients match exactly.

```bash
cd step_03_backprop && cargo run
```

---

## 🔘 Step 4: One Brain Cell (The Neuron)

### 💡 The Analogy
A **neuron** takes several inputs, multiplies each by a **weight**, adds a **bias**, and squashes
the result with `tanh`. That's it. It's a weighted vote followed by a "how excited am I?" knob.

```bash
cd step_04_neuron && cargo run
```

---

## 🧱 Step 5: Stacking Cells Into a Brain (Layers & the MLP)

### 💡 The Analogy
One neuron is weak. A **Layer** is a row of neurons; an **MLP** (Multi-Layer Perceptron) is several
layers stacked. We wrap them behind a `Module` **trait** — the exact interface Burn's
`#[derive(Module)]` gives the transformer.

```bash
cd step_05_layer_mlp && cargo run
```

---

## 🎯 Step 6: The Scorecard (Loss)

### 💡 The Analogy
How wrong is the model *right now*? The **loss** is a single number measuring total mistake.
We use **Mean Squared Error**: average of (guess − answer)², the same shape as the transformer's
cross-entropy scorecard.

```bash
cd step_06_loss && cargo run
```

---

## ⛷️ Step 7: Walking Downhill By Hand (Gradient Descent)

### 💡 The Analogy
The loss is a hill; we want the bottom. `backward()` tells us which way is downhill for every
weight (its `grad`). We take a small step the *opposite* way: `weight -= learning_rate * grad`.
Repeat, and watch the loss fall.

```bash
cd step_07_gradient_descent && cargo run
```

---

## 🏎️ Step 8: Building Adam (The Optimizer)

### 💡 The Analogy
Plain gradient descent is a hiker taking equal steps. **Adam** is a smart hiker with *momentum*
(keep rolling in a good direction) and *adaptive step sizes* (tiptoe on steep ground, stride on
flat ground). This is the `AdamConfig` the transformer used — now built by hand.

```bash
cd step_08_optimizer && cargo run
```

---

## 🔁 Step 9: The Tutor (The Training Loop)

### 💡 The Analogy
Put it all together into the rhythm every model trains by:
**forward → loss → zero grads → backward → optimizer step**, repeated each epoch.
This is exactly what Burn's `Learner` does for the transformer — minus the magic.

```bash
cd step_09_train_loop && cargo run
```

---

## 🏁 Step 10: Train a Real Net (The Tiny MLP)

### 💡 The Analogy
Graduation! We make a 2D dataset (dots inside a circle vs. outside), train an MLP to tell them
apart, and **draw the decision boundary in ASCII**. A complete neural network, trained by an
engine you wrote from nothing.

```bash
cd step_10_tiny_mlp && cargo run
```

---

## 🏆 What Next?
Reopen `handcrafted_transformer/step_10_complete_gpt/src/training.rs` and read
`out.loss.backward()` again. It's the same engine you just built — only with tensors instead of
scalars, and a million parameters instead of a few dozen. No more magic. 🧠
