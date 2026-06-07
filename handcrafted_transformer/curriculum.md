# 🚀 Handcrafted Transformer: 10-Step Journey to Building Your Own AI

Welcome, Future AI Engineer! 

Have you ever wondered how ChatGPT is able to read your questions and write back like a human? Inside ChatGPT is a special math engine called a **Transformer**. 

In this journey, we are going to build a mini-Transformer from scratch in Rust! We will start with the absolute basics (turning words into numbers) and finish with a model that can write its own stories.

**Each step is its own independent project folder.** You can `cd` into any step and run `cargo run` to see it in action!

---

## 🗺️ The Map of Our Journey

```
step_01_tokenizer/             ➔ Turn words into secret code numbers
       │
step_02_dataset/               ➔ Make practice flashcards from a book
       │
step_03_embedding/             ➔ Place words on a 2D map (GPS coordinates)
       │
step_04_positional_encoding/   ➔ Add page numbers so order matters
       │
step_05_attention/             ➔ Shine a flashlight on related words
       │
step_06_causal_masking/        ➔ Cover the future (no cheating!)
       │
step_07_multi_head_attention/  ➔ Form a team of specialized detectives
       │
step_08_feed_forward/          ➔ Add a brainy decision-making layer
       │
step_09_transformer_block/     ➔ Build a factory assembly line
       │
step_10_complete_gpt/          ➔ Practice with a tutor & write stories!
```

---

## 🕵️‍♂️ Step 1: The Secret Decoder Ring (The Tokenizer)

### 💡 The Analogy
Computers are actually giant calculators—they don't understand letters or words, only numbers! A **Tokenizer** is like a **Secret Decoder Ring**. It takes a sentence, cleans up the punctuation, splits it into words, and assigns a unique number code to every word it has ever seen.

### 🎮 Run the Demo!
```bash
cd step_01_tokenizer && cargo run
```

---

## 🗂️ Step 2: Training Flashcards (The Dataset)

### 💡 The Analogy
How do you study for a spelling test? You use **flashcards**! On the front, you have a clue, and on the back, you have the answer. 
To train our AI, we make thousands of sliding-window flashcards from a book. Every target word is just the input sequence shifted by one word into the future!

### 🎮 Run the Demo!
```bash
cd step_02_dataset && cargo run
```

---

## 📍 Step 3: Words on a GPS Map (Embeddings)

### 💡 The Analogy
If you just give words random numbers, the computer doesn't know that *"apple"* and *"banana"* are both fruits.
An **Embedding** turns each number code into **GPS coordinates on a map** where similar words are placed close together! We use the Pythagorean theorem to measure distances.

### 🎮 Run the Demo!
```bash
cd step_03_embedding && cargo run
```

---

## 📖 Step 4: Adding Page Numbers (Positional Encoding)

### 💡 The Analogy
Without order, *"the dog bit the boy"* and *"the boy bit the dog"* look exactly the same to the model.
We add **Positional Encodings**—smooth mathematical waves (sines and cosines) that create a unique fingerprint for every position in a sentence.

### 🎮 Run the Demo!
```bash
cd step_04_positional_encoding && cargo run
```

---

## 🔦 Step 5: Focusing Your Flashlight (Single-Head Attention)

### 💡 The Analogy
When you read the word **"bank"** in *"I sat by the river bank"*, how do you know it's not a money bank? You look at the word **"river"**!
**Attention** is like shining a flashlight. We use dot products to find how well words match, and softmax to turn scores into percentages.

### 🎮 Run the Demo!
```bash
cd step_05_attention && cargo run
```

---

## 🚫 Step 6: No Cheating! (Causal Masking)

### 💡 The Analogy
If you are taking a next-word prediction test, you can't look at the answer page! 
We use a **Causal Mask** to block the AI from looking into the future. By adding negative infinity (−1,000,000,000) to future attention scores, their probability becomes exactly 0%.

### 🎮 Run the Demo!
```bash
cd step_06_causal_masking && cargo run
```

---

## 👥 Step 7: A Team of Detectives (Multi-Head Attention)

### 💡 The Analogy
If you only have one flashlight, you can only focus on one thing. So we build a **team of detectives** (called **heads**). Each detective works independently in their own room, then they gather to combine their notes into one master report.

### 🎮 Run the Demo!
```bash
cd step_07_multi_head_attention && cargo run
```

---

## 🧠 Step 8: The Brain's Decision Filter (Feed-Forward & GELU)

### 💡 The Analogy
After our detectives gather their clues, the AI needs to *think* and make a decision. The **Feed-Forward Network** with **GELU** activation acts like a dimmer switch: if a number is negative (unimportant), it dims it to zero. If it's positive, it lets it pass through. This is where the model gains the power to learn complex rules.

### 🎮 Run the Demo!
```bash
cd step_08_feed_forward && cargo run
```

---

## 🏗️ Step 9: The Factory Assembly Line (The Transformer Block)

### 💡 The Analogy
A **Transformer Block** is a single station in our factory:
1. Normalize (LayerNorm)
2. Inspect context (Attention) + keep the blueprint (Residual Connection)
3. Normalize again
4. Think and filter (Feed-Forward) + keep the blueprint again

By linking multiple blocks together, the model gets smarter at each station!

### 🎮 Run the Demo!
```bash
cd step_09_transformer_block && cargo run
```

---

## 🎓 Step 10: Training & Storytime (The Complete GPT Model)

### 💡 The Analogy
Now, it's school time! We assemble our complete model:
* **The Student**: The Transformer.
* **The Tutor**: The training loop.
* **The Scorecard**: The Loss Function (Cross-Entropy).
* **The Coach**: The Optimizer (Adam).

We will train on a tiny text corpus, watch the loss score drop, and then ask the model to finish prompts!

### 🎮 Run the Demo!
```bash
cd step_10_complete_gpt
cargo run --release --bin train      # Train the model
cargo run --release --bin generate   # Generate stories!
```

---

## 🏆 What Next?
Once you understand these 10 steps, you understand the core math of modern AI! Each step folder has its own `README.md` with deeper explanations and middle-school-level math breakdowns.

Happy coding! 🚀
