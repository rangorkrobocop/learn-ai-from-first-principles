# Step 3: Pouring Gradients Backwards (Backpropagation)

This is the heart of the whole engine. After this step, **everything else is just built on top.**

## 💡 The Analogy
Water flows downhill; gradients flow **backwards**. We seed the final answer with a slope of `1.0`
(it affects itself one-for-one), then walk the graph in reverse. At each node, the **chain rule**
multiplies slopes together and hands the gradient back to its parents.

## 📐 Basic Mathematics: The Chain Rule
If `c` depends on `b`, and `b` depends on `a`, then:
$$\frac{dc}{da} = \frac{dc}{db} \times \frac{db}{da}$$
"How `a` affects `c`" = "how `b` affects `c`" × "how `a` affects `b`."

Each operation knows its own little derivative:
- `a + b` → gradient passes **straight through** to both parents.
- `a * b` → each parent scales by the **other's** value.
- `tanh(x)` → multiply by `1 - tanh(x)²`.

## 🧪 The demo
We rebuild Andrej Karpathy's famous micrograd neuron —
`o = tanh(x1*w1 + x2*w2 + b)` — call `o.backward()` once, and check the gradients match the
known-correct values exactly (`x1.grad = -1.5`, `w1.grad = +1.0`, …). The program `assert!`s the
check passes.

## 🔗 Burn bridge
This `backward()` is the scalar twin of `out.loss.backward()` in
`handcrafted_transformer/step_10_complete_gpt/src/training.rs`. Same algorithm — topological sort
then reverse chain-rule — just one number at a time instead of whole tensors.

## 🚀 How to Run
```bash
cargo run
```
Runs the forward pass, calls `backward()`, prints every gradient, and verifies them against the math.

> 📌 The engine in this file (`Value`, `backward`, `tanh`, `relu`, `powf`, the operators) is the
> **complete** engine. Steps 4–10 reuse it verbatim and build neurons, layers, loss, and Adam on top.
