# Step 3: Words as GPS Coordinates on a Map (Embeddings)

Instead of looking at words as arbitrary number codes, we want the AI to understand that some words have similar meanings. We do this by giving each word its own GPS coordinates.

## 💡 The Analogy
Imagine a map of the world. If you place foods in one country, sports in another, and electronics in a third:
* `"apple"` is close to `"banana"`
* `"apple"` is very far from `"laptop"`

An **Embedding** is a lookup table that converts a word ID into these coordinates.

## 📐 Middle School Math: Pythagorean Theorem
To measure how similar two words are, we find the straight-line distance between their coordinates.
$$Distance = \sqrt{(x_2 - x_1)^2 + (y_2 - y_1)^2}$$
If the distance is small, the words are similar. If it is large, they are different!

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Initialize a random 2D coordinate lookup table using the Burn deep learning framework.
2. Look up the coordinates for `"cat"`, `"dog"`, and `"computer"`.
3. Compute the distances between them using the Pythagorean theorem.
