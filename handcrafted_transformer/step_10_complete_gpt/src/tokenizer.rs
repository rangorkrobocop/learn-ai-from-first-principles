/// Word-level tokenizer with a fixed vocabulary built from the training corpus.
/// This is intentionally simple so you can see exactly what tokenization does:
/// every unique whitespace-split word gets an integer ID, plus special tokens.
use std::collections::HashMap;

pub const PAD_TOKEN: &str = "<pad>";
pub const UNK_TOKEN: &str = "<unk>";
pub const BOS_TOKEN: &str = "<bos>";
pub const EOS_TOKEN: &str = "<eos>";

#[derive(Debug, Clone)]
pub struct Tokenizer {
    /// word → integer id
    pub word_to_id: HashMap<String, usize>,
    /// integer id → word
    pub id_to_word: Vec<String>,
}

impl Tokenizer {
    /// Build vocabulary from raw text.  
    /// We lowercase everything and split on whitespace + punctuation.
    pub fn build_from_text(text: &str) -> Self {
        let mut word_to_id: HashMap<String, usize> = HashMap::new();
        let mut id_to_word: Vec<String> = Vec::new();

        // Insert special tokens first so they always have low IDs.
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

        Tokenizer {
            word_to_id,
            id_to_word,
        }
    }

    pub fn vocab_size(&self) -> usize {
        self.id_to_word.len()
    }

    pub fn pad_id(&self) -> usize {
        self.word_to_id[PAD_TOKEN]
    }

    pub fn bos_id(&self) -> usize {
        self.word_to_id[BOS_TOKEN]
    }

    pub fn eos_id(&self) -> usize {
        self.word_to_id[EOS_TOKEN]
    }

    pub fn unk_id(&self) -> usize {
        self.word_to_id[UNK_TOKEN]
    }

    /// Encode a string into a sequence of token IDs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        tokenize_raw(text)
            .into_iter()
            .map(|w| {
                *self
                    .word_to_id
                    .get(&w)
                    .unwrap_or(&self.word_to_id[UNK_TOKEN])
            })
            .collect()
    }

    /// Decode a sequence of token IDs back to a string.
    pub fn decode(&self, ids: &[usize]) -> String {
        ids.iter()
            .filter(|&&id| id != self.pad_id() && id != self.bos_id() && id != self.eos_id())
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

/// Split raw text into lowercase words, stripping punctuation.
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
