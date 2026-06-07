/// Step 1: The Secret Decoder Ring (The Tokenizer)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   Computers are actually giant calculators—they don't understand letters or words,
///   only numbers! A Tokenizer is like a Secret Decoder Ring. It takes a sentence,
///   cleans up the punctuation, splits it into words, and assigns a unique number
///   code to every word it has ever seen.

use std::collections::HashMap;

pub const PAD_TOKEN: &str = "<pad>"; // Filler/padding for short sentences
pub const UNK_TOKEN: &str = "<unk>"; // For words we've never seen before
pub const BOS_TOKEN: &str = "<bos>"; // "Beginning Of Sentence" (start marker)
pub const EOS_TOKEN: &str = "<eos>"; // "End Of Sentence" (stop marker)

#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// maps: word ➔ number code
    pub word_to_id: HashMap<String, usize>,
    /// maps: number code ➔ word
    pub id_to_word: Vec<String>,
}

impl Tokenizer {
    /// Build our secret decoder ring from a sample text (our "book" or "corpus")
    pub fn build_from_text(text: &str) -> Self {
        let mut word_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        // 1. Insert special tokens first so they get the lowest codes (0, 1, 2, 3)
        for special in [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN] {
            word_to_id.insert(special.to_string(), id_to_word.len());
            id_to_word.push(special.to_string());
        }

        // 2. Clean and split the text into words, then assign codes
        for word in tokenize_raw(text) {
            if !word_to_id.contains_key(&word) {
                word_to_id.insert(word.clone(), id_to_word.len());
                id_to_word.push(word);
            }
        }

        Tokenizer {
            word_to_id,
            id_to_word,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn unk_id(&self) -> usize {
        self.word_to_id[UNK_TOKEN]
    }

    /// Turn a sentence into a list of number codes
    pub fn encode(&self, text: &str) -> Vec<usize> {
        tokenize_raw(text)
            .into_iter()
            .map(|w| {
                // Look up the word. If we don't know it, use the <unk> (unknown) code!
                *self.word_to_id.get(&w).unwrap_or(&self.word_to_id[UNK_TOKEN])
            })
            .collect()
    }

    /// Turn a list of number codes back into a sentence
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            // Ignore special start/stop/filler codes when converting back to text
            .filter(|&&id| {
                id != self.word_to_id[PAD_TOKEN]
                    && id != self.word_to_id[BOS_TOKEN]
                    && id != self.word_to_id[EOS_TOKEN]
            })
            .map(|&id| {
                self.id_to_word
                    .get(id)
                    .map(|s| s.as_str())
                    .unwrap_or(UNK_TOKEN)
            })
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Helper function: Splits text by space, lowercases it, and strips punctuation.
/// e.g. "Hello, World!" ➔ ["hello", "world"]
fn tokenize_raw(text: &str) -> Vec<String> {
    text.split_whitespace()
        .map(|w| {
            w.to_lowercase()
                .chars()
                .filter(|c| c.is_alphabetic())
                .collect::<String>()
        })
        .filter(|w| !w.is_empty())
        .collect()
}

fn main() {
    println!("🕵️‍♂️ STEP 1: THE SECRET DECODER RING (THE TOKENIZER)");
    println!("==================================================");
    println!("Welcome to Step 1! Since computers don't understand words,");
    println!("we must turn words into numbers. Let's build a decoder ring!");
    println!();

    // 1. Build a mini corpus (our training text)
    let training_corpus = "The cat sat on the mat. A robot sat on the computer.";
    println!("📖 Step 1.1: Building a Secret Ring from this text:");
    println!("   \"{}\"", training_corpus);
    println!();

    // 2. Build the tokenizer
    let tokenizer = Tokenizer::build_from_text(training_corpus);

    println!("📚 Step 1.2: The Decoder Ring has learned {} words!", tokenizer.vocab_size());
    println!("Here is the word-to-number dictionary:");
    for (id, word) in tokenizer.id_to_word.iter().enumerate() {
        println!("  ID {:>2} ➔ \"{}\"", id, word);
    }
    println!();

    // 3. Tokenize a new sentence
    let test_sentence = "the robot sat on the mat";
    println!("📝 Step 1.3: Encoding a new sentence:");
    println!("  Input text:  \"{}\"", test_sentence);
    
    let encoded = tokenizer.encode(test_sentence);
    println!("  Encoded IDs: {:?}", encoded);
    println!();

    // 4. Decode it back
    let decoded = tokenizer.decode(&encoded);
    println!("🔄 Step 1.4: Decoding it back to words:");
    println!("  Decoded text: \"{}\"", decoded);
    println!();

    // 5. Show unknown token handling
    let strange_sentence = "the dinosaur sat on the computer";
    println!("🦖 Step 1.5: What happens to words the ring has never seen?");
    println!("  Input text:  \"{}\"", strange_sentence);
    
    let strange_encoded = tokenizer.encode(strange_sentence);
    println!("  Encoded IDs: {:?}", strange_encoded);
    println!("  (Note that 'dinosaur' gets turned into code {}, which is <unk> for 'unknown'!)", tokenizer.unk_id());
    println!();
    println!("🎉 Step 1 Complete! You now know how models translate words into number codes.");
}
