# Step 10: Training & Storytime (The Complete GPT Model)

This is the final step! We assemble everything from Steps 1–9 into a complete GPT-style Transformer, train it on a small text corpus, and then generate new text.

## 🎓 The Analogy
Now, it's school time! We assemble our complete learning system:
* **The Student**: The Transformer model (Embedding + Positional Encoding + N × Transformer Blocks + Output Head).
* **The Textbook**: Our training corpus (`data/corpus.txt`) — chopped into flashcards.
* **The Tutor**: The training loop (shows flashcards, checks answers, and corrects mistakes).
* **The Scorecard**: The **Loss Function** (Cross-Entropy Loss) — tells us how wrong the student is.
* **The Coach**: The **Optimizer** (Adam) — tells the student exactly how to adjust all its coordinate maps, detective settings, and brain filters to improve.

We train for several epochs (study sessions), watch the loss drop, and then ask the model to continue sentences!

## 📐 Middle School Math: Cross-Entropy Loss
After the model predicts probabilities for each next word (using softmax), we check: *"How much probability did you assign to the correct answer?"*
* If the model gave 90% to the correct word, the loss is **low** (good!).
* If the model gave 1% to the correct word, the loss is **high** (bad!).

The formula uses logarithms:
$$\text{Loss} = -\log(\text{probability of correct word})$$

The optimizer then nudges all the weights in a direction that increases the probability of correct answers.

## 🚀 How to Run

### Train the model:
```bash
cargo run --release --bin train
```
This reads `data/corpus.txt`, trains for 30 epochs, and saves the model to `artifacts/`.

### Generate text:
```bash
cargo run --release --bin generate
```
This loads the trained model and generates text using three strategies:
1. **Greedy** — always picks the most likely next word.
2. **Temperature sampling** — adds randomness for creativity.
3. **Top-K sampling** — picks randomly from only the top K most likely words.

## 📁 Project Structure
```
step_10_complete_gpt/
├── Cargo.toml           # Project config
├── data/
│   └── corpus.txt       # Training text
├── src/
│   ├── lib.rs           # Module declarations
│   ├── tokenizer.rs     # Step 1: Words → Numbers
│   ├── dataset.rs       # Step 2: Sliding-window flashcards
│   ├── model.rs         # Steps 3–9: Full transformer architecture
│   ├── training.rs      # Training loop with Burn
│   ├── generation.rs    # Text generation strategies
│   └── bin/
│       ├── train.rs     # Training entry point
│       └── generate.rs  # Generation entry point
└── README.md
```
