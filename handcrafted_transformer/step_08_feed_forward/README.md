# Step 8: The Brain's Decision Filter (Feed-Forward & GELU)

Attention layers only blend and average word coordinates. They do not perform complex computations. The **Feed-Forward Network (FFN)** is the actual brainy layer where decisions are made.

## 🧠 The Analogy
Imagine we have detectives gathering clues (Attention). That's great, but they need to think!
The **Feed-Forward Network** processes the clues for each word independently. Inside it, it uses a non-linear activation curve called **GELU** (Gaussian Error Linear Unit). 

GELU acts like a **dimmer switch** or filter for electrical signals:
* If the incoming signal is negative (useless noise), it dims it down to exactly `0`.
* If the incoming signal is positive (important information), it lets it pass through.

## 📐 Middle School Math: Why Non-Linearity Matters
If we only used linear math (multiplications and additions), a 100-layer neural network would behave exactly like a 1-layer network.
By introducing a "kink" or a curve that squishes some values to 0 (non-linearity), we allow the neural network to build complex logic gates, like:
* *If the word is `"bank"` AND it is next to `"river"`, then turn off the money meaning, and turn on the water meaning.*

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Pass numbers from `-3.0` to `3.0` through the GELU activation function and print the inputs and outputs, showing how it blocks negative numbers.
2. Initialize a Feed-Forward Network layer in Burn.
3. Pass a mock tensor through the layer and display its input/output shapes.
