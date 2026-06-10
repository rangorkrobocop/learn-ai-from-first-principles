# Step 6: No Cheating! (Causal Masking)

To train an AI, we show it a whole sentence. But when it tries to predict word number 3, we must hide word number 4, 5, etc., from it. Otherwise, it would just copy the answers instead of learning to predict!

## 💡 The Analogy
Imagine reading a murder mystery book. You want to guess the culprit on page 50 using only the clues up to page 50. If you peek at page 100, you cheat!
A **Causal Mask** is like holding a cardboard screen that slides across the sentence from left to right as the model reads.

## 📐 Basic Mathematics: Adding Negative Infinity
How do we mathematically force a probability to be $0\%$? 
We add a huge negative number like $-1,000,000,000$ (negative one billion, standing in for negative infinity $-\infty$) to the scores of all future words.

When we compute **Softmax** (converting scores to percentages):
$$e^{-1,000,000,000} \approx 0$$
So the weight of future words becomes exactly $0\%$.

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Create a causal mask matrix of shape `[4, 4]`.
2. Generate raw attention scores.
3. Apply the causal mask by adding it to the raw scores.
4. Calculate the softmax percentages and print the final triangular attention grid, showing that words can never look at future words.
