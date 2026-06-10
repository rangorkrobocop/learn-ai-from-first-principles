# Step 7: A Team of Detectives (Multi-Head Attention)

In human language, a word can be related to others in many different ways at the same time. To capture this complexity, we use a team of attention flashlights instead of just one.

## 👥 The Analogy
Imagine writing a report on a book. If you do it alone, you might miss things. Instead, you hire a **team of detectives** (called **heads**):
* **Detective 1**: Focuses on verbs (who does what?).
* **Detective 2**: Focuses on descriptions (how does it look?).

Each detective splits up the word coordinates, does attention in their own room, and then they gather in the meeting room, paste their notes together (concatenation), and write a combined report.

## 📐 Basic Mathematics: Dividing and Joining Coordinates
If our word coordinates have 4 numbers:
$$[x_1, x_2, x_3, x_4]$$
And we have 2 heads (detectives):
* **Detective 1** gets the first 2 numbers: $[x_1, x_2]$
* **Detective 2** gets the last 2 numbers: $[x_3, x_4]$

They do their calculations independently. When done, they join their answers back together:
$$[y_1, y_2] \text{ and } [y_3, y_4] \rightarrow [y_1, y_2, y_3, y_4]$$

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Initialize a `MultiHeadSelfAttention` layer with 2 heads using Burn.
2. Feed a mock sentence through.
3. Trace and print out the shapes of the tensors as they are split apart and merged back together.
