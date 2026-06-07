/// Step 2: Training Flashcards (The Dataset)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   How do you study for a vocabulary test? You use flashcards! On the front,
///   you have a clue, and on the back, you have the answer.
///   To train our AI to predict the next word, we make thousands of sliding-window
///   flashcards from a book. If the text is "the cat sat on the mat", and our card
///   can hold 3 words, our flashcards look like this:
///     * Card 1: Input (Front): "the cat sat" ➔ Target (Back): "cat sat on"
///
///   Notice that the target is just the input shifted by one word into the future!
///   This teaches the AI:
///     - Given "the", predict "cat"
///     - Given "the cat", predict "sat"
///     - Given "the cat sat", predict "on"

use std::collections::HashMap;

// ── Re-using Tokenizer from Step 1 so this project runs independently ────────
pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

#[derive(Debug, Clone)]
pub struct Tokenizer {
    pub word_to_id: HashMap<String, usize>,
    pub id_to_word: Vec<String>,
}

impl Tokenizer {
    pub fn build_from_text(text: &str) -> Self {
        let mut word_to_id = HashMap::new();
        let mut id_to_word = Vec::new();
        for special in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] {
            word_to_id.insert(special.to_string(), id_to_word.len());
            id_to_word.push(special.to_string());
        }
        for word in tokenize_raw(text) {
            if !word_to_id.contains_key(&word) {
                word_to_id.insert(word.clone(), id_to_word.len());
                id_to_word.push(word);
            }
        }
        Tokenizer { word_to_id, id_to_word }
    }

    pub fn encode(&self, text: &str) -> Vec<usize> {
        tokenize_raw(text)
            .into_iter()
            .map(|w| *self.word_to_id.get(&w).unwrap_or(&self.word_to_id[UNK_TOKEN]))
            .collect()
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.word_to_id[PAD_TOKEN] && id != self.word_to_id[BOS_TOKEN] && id != self.word_to_id[EOS_TOKEN])
            .map(|&id| self.id_to_word.get(id).map(|s| s.as_str()).unwrap_or(UNK_TOKEN))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

fn tokenize_raw(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| w.to_lowercase().chars().filter(|c| c.is_alphabetic()).collect())
        .filter(|w: &String| !w.is_empty())
        .collect()
}

// ── Dataset Implementation ───────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LmItem {
    pub input: Vec<usize>,  // Front of the flashcard (Input IDs)
    pub target: Vec<usize>, // Back of the flashcard (Target IDs)
}

pub struct LmDataset {
    pub items: Vec<LmItem>,
}

impl LmDataset {
    /// Create sliding-window flashcards of size `seq_len` from our corpus
    pub fn new(corpus: &str, tokenizer: &Tokenizer, seq_len: usize) -> Self {
        let tokens = tokenizer.encode(corpus);
        let n = tokens.len();
        let mut items = Vec::new();

        // We need seq_len + 1 tokens to build one input-target pair.
        // For example, if seq_len is 3, we need 4 tokens:
        // Tokens: [A, B, C, D]
        // Input:  [A, B, C]
        // Target: [B, C, D]
        for start in 0..(n.saturating_sub(seq_len)) {
            let input = tokens[start..start + seq_len].to_vec();
            let target = tokens[start + 1..start + seq_len + 1].to_vec();
            items.push(LmItem { input, target });
        }

        LmDataset { items }
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }
}

fn main() {
    println!("🗂️ STEP 2: TRAINING FLASHCARDS (THE DATASET)");
    println!("============================================");
    println!("Welcome to Step 2! We will now turn a book into sliding-window");
    println!("flashcards, showing the AI what word comes next.");
    println!();

    // 1. Setup tokenizer
    let corpus = "the robot sat on the computer and built a transformer";
    let tokenizer = Tokenizer::build_from_text(corpus);

    // 2. Setup Dataset with sliding window length = 3
    let seq_len = 3;
    println!("📖 Step 2.1: Reading our tiny book:");
    println!("   \"{}\"", corpus);
    println!();
    println!("   We will use a flashcard length (sequence length) of {} words.", seq_len);
    println!();

    let dataset = LmDataset::new(corpus, &tokenizer, seq_len);
    println!("📚 Step 2.2: Created {} flashcards!", dataset.len());
    println!();

    // 3. Print out each flashcard
    println!("🔍 Step 2.3: Let's inspect the front and back of each card:");
    for i in 0..dataset.len() {
        let item = &dataset.items[i];
        let input_words = tokenizer.decode(&item.input);
        let target_words = tokenizer.decode(&item.target);
        
        println!("🎴 Flashcard #{} :", i + 1);
        println!("   [Front - Input IDs]    {:?} ➔ Words: \"{}\"", item.input, input_words);
        println!("   [Back  - Target IDs]   {:?} ➔ Words: \"{}\"", item.target, target_words);
        println!("   (Prediction Goal: Given \"{}\", predict \"{}\")", input_words, target_words);
        println!();
    }

    println!("🎉 Step 2 Complete! You now understand how we format data to train an AI.");
}
