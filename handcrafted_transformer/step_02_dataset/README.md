# Step 2: Training Flashcards (The Dataset)

To train an AI, we cannot just give it a book and say "read this". We must break the book into thousands of study questions, or **flashcards**.

## 💡 The Analogy
Imagine training a dog. You don't teach it a whole routine at once. You teach it transitions:
* If I say `"sit"`, you do `"stay"`.
* If I say `"stay"`, you do `"come"`.

In language modelling:
* If the input is `"the robot sat"`, the target we want the model to output is `"robot sat on"`.
* Notice how the target is the exact same sequence shifted by one word into the future. This means at each word, the model tries to guess the very next word!

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Load a simple sentence (our "book").
2. Create flashcards using a sequence window of 3 words.
3. Show you the "Front" (what the AI reads) and "Back" (the correct answers the AI is checked against) of every flashcard.
