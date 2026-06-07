# Step 1: The Secret Decoder Ring (The Tokenizer)

Computers only understand numbers, not words. A **Tokenizer** acts like a secret decoder ring that assigns a unique number ID to every word.

## 💡 The Analogy
Imagine you want to send a secret message to a friend. You agree that:
* `"the"` is number `4`
* `"robot"` is number `5`
* `"sat"` is number `6`

So instead of writing `"the robot sat"`, you write `[4, 5, 6]`. That's tokenization!

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Learn a small vocabulary of words from a sentence.
2. Show you the dictionary it made.
3. Translate a new sentence into a list of numbers (Encoding).
4. Translate it back to words (Decoding).
5. Show how it handles a word it has never seen before using the special `<unk>` (unknown) sticker.
