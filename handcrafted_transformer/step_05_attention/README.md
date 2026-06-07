# Step 5: Focusing Your Flashlight (Single-Head Attention)

Attention is the core secret of modern AI. It allows a word to look at other words in a sentence to get their context, like a reader looking back at earlier words.

## 🔦 The Analogy: Query, Key, Value
Imagine a filing cabinet:
1. **Query (Q)**: The sticky note you hold, saying: *"I am looking for water-related words."*
2. **Key (K)**: The labels on the folders in the cabinet, saying: *"I have water words!"* or *"I have crime words!"*
3. **Value (V)**: The actual documents inside the folders, containing the meaning.

We compare the **Query** with all the **Keys** to get a match score. Then we use those scores to blend the **Values** together.

## 📐 Middle School Math: Dot Product & Softmax
To find out if two coordinate arrows point in the same direction, we use the **Dot Product**:
$$Score = (Q_x \times K_x) + (Q_y \times K_y)$$
If they point in the same direction, the score is high. If they are perpendicular, the score is 0. If they point in opposite directions, the score is negative.

Then, we use **Softmax** (a percentage split) to convert these scores into percentages that sum to 100%.

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Set up key coordinates for the words: `"the"`, `"river"`, `"bank"`, `"robber"`.
2. Let the word `"bank"` ask a water-context query.
3. Compute the dot-product scores and run them through softmax.
4. Draw an ASCII flashlight beam showing how much percentage of attention is directed to each word!
