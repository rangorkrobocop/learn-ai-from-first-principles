# Step 4: Adding Page Numbers (Positional Encoding)

In a sentence, word order matters. But basic attention layers process words all at once (like a bag of words). To fix this, we add sequence position information.

## 💡 The Analogy
Imagine we print out a dictionary, but we scramble the page order. It's hard to read! 
To keep order, we write **page numbers** or **stickers** directly on the word flashcards before we read them. 

Instead of adding raw numbers (like 1, 2, 3) which can grow too large and confuse the neural network, we use repeating wave patterns (**sines and cosines**) to assign a unique coordinate sticker between `-1.0` and `1.0` to each position.

## 🌊 Basic Mathematics: Sine and Cosine Waves
Think of a swing going back and forth:
* At time 0, it is in the middle (0)
* At time 1, it swings right (1)
* At time 2, it swings back to the middle (0)
* At time 3, it swings left (-1)

By combining many swings (waves) of different speeds, we can give every position index in a sentence a unique coordinate pattern.

## 🚀 How to Run
In this folder, run:
```bash
cargo run
```
This will:
1. Generate wave-based positional encoding stickers for Positions 0, 1, and 2.
2. Show you the coordinate values of these position stickers.
3. Take a word like `"cat"` and show how its coordinate representation shifts depending on where it sits in a sentence.
