# Step 2: Remembering Your Parents (Operations & the Graph)

A result should remember **how it was made**. We teach `+` and `*` to record their parents.

## 💡 The Analogy
When you compute `e = a * b`, the answer `e` should remember it was born from `a` **times** `b`.
So every `Value` produced by an operation now stores:
- `prev` — the **parent** values that made it, and
- `op` — **which** operation made it (`"+"`, `"*"`, …).

Chain many operations together and you've drawn a family tree — the **computation graph**.

## 🌳 Forward vs. backward
- Follow the arrows **down** (parents → child) to compute the answer. That's the **forward pass**.
- In Step 3 we'll follow them back **up** (child → parents) to compute slopes. That's **backprop**.

The demo builds `d = a * b + c` and prints the whole graph underneath `d`.

## 🦀 Rust note
We implement Rust's `Add` and `Mul` **traits** for `&Value`, so you can write `&a * &b` and get a
new `Value` that records its parents. (A `trait` is Rust's word for a shared interface — the same
mechanism Burn uses for `Module`.)

## 🔗 Burn bridge
Burn builds this exact graph automatically for the transformer every time you write `matmul`,
`+`, `softmax`, etc. — it has to, so `loss.backward()` can find the path back to every weight.

## 🚀 How to Run
```bash
cargo run
```
Builds `d = a * b + c` and prints its computation graph.
