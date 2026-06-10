# Step 9: The Factory Assembly Line (The Transformer Block)

A **Transformer Block** combines everything from Steps 5–8 into one reusable station on a factory assembly line.

## 🏗️ The Analogy
In a car factory, the car moves along a **conveyor belt** through different stations:
1. **Wash** (LayerNorm) — Standardize the incoming parts so they are not too big or too small.
2. **Inspect** (Multi-Head Attention) — Detectives look at neighboring words for context.
3. **Keep the blueprint** (Residual Connection) — Add the attention changes BACK to the original input, so we never lose track of the original words.
4. **Wash again** (LayerNorm).
5. **Think and filter** (Feed-Forward + GELU) — Process the features independently for each word.
6. **Keep the blueprint again** (Residual Connection).

By linking multiple blocks together (stacking them), the model gets smarter and smarter at each station!

## 📐 Basic Mathematics: Residual Connections
The key trick that makes deep networks work is:
$$\text{output} = x + \text{SubLayer}(x)$$
Instead of *replacing* the input with the result, we *add* the result to the original. This way:
* The original signal always passes through (like a shortcut highway).
* The sub-layer only needs to learn the *difference* (what to add), not the whole answer from scratch.
* Gradients can flow backwards through the addition without shrinking (vanishing).

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Build a complete `TransformerBlock` with LayerNorm, Multi-Head Attention, and Feed-Forward sub-layers.
2. Pass a mock 3-word sentence through the block.
3. Print the input and output tensor shapes.
