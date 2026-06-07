/// Language modelling dataset — sliding-window next-token-prediction.
///
/// Given a token sequence of length N, we create windows of size seq_len+1:
///   input  = window[0..seq_len]
///   target = window[1..seq_len+1]
///
/// Burn 0.17+ changed the Batcher trait:
///   - Now `Batcher<B, Item, Batch>` (B is the backend, first type param)
///   - `fn batch(&self, items: Vec<I>, device: &B::Device) -> O`
///     (device is passed in, not stored in the batcher)
use burn::{
    data::{
        dataloader::batcher::Batcher,
        dataset::Dataset,
    },
    prelude::*,
};

use crate::tokenizer::Tokenizer;

// ── Raw dataset item ─────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LmItem {
    pub input:  Vec<usize>,  // [seq_len]
    pub target: Vec<usize>,  // [seq_len]  (input shifted right by 1)
}

// ── Dataset ──────────────────────────────────────────────────────────────────

pub struct LmDataset {
    items: Vec<LmItem>,
}

impl LmDataset {
    pub fn new(corpus: &str, tokenizer: &Tokenizer, seq_len: usize) -> Self {
        let tokens = tokenizer.encode(corpus);
        let n = tokens.len();
        let mut items = Vec::new();
        for start in 0..(n.saturating_sub(seq_len)) {
            items.push(LmItem {
                input:  tokens[start..start + seq_len].to_vec(),
                target: tokens[start + 1..start + seq_len + 1].to_vec(),
            });
        }
        println!(
            "[dataset] corpus tokens: {}  windows: {}  seq_len: {}",
            n, items.len(), seq_len
        );
        LmDataset { items }
    }
}

impl Dataset<LmItem> for LmDataset {
    fn get(&self, index: usize) -> Option<LmItem> { self.items.get(index).cloned() }
    fn len(&self) -> usize { self.items.len() }
}

// ── Tensor batch ─────────────────────────────────────────────────────────────

#[derive(Clone, Debug)]
pub struct LmBatch<B: Backend> {
    pub input:  Tensor<B, 2, Int>,   // [batch, seq_len]
    pub target: Tensor<B, 2, Int>,   // [batch, seq_len]
}

// ── Batcher — Burn 0.17+ signature ───────────────────────────────────────────
//
// Trait: Batcher<B, I, O>
//   B = Backend
//   I = input item type
//   O = output batch type
//
// fn batch(&self, items: Vec<I>, device: &B::Device) -> O
//
// The batcher itself stores NO device; the device is injected at call time
// by the DataLoader (which is now generic over B).

#[derive(Clone, Default)]
pub struct LmBatcher;

impl<B: Backend> Batcher<B, LmItem, LmBatch<B>> for LmBatcher {
    fn batch(&self, items: Vec<LmItem>, device: &B::Device) -> LmBatch<B> {
        let batch_size = items.len();
        let seq_len = items[0].input.len();

        let mut inp_flat = Vec::with_capacity(batch_size * seq_len);
        let mut tgt_flat = Vec::with_capacity(batch_size * seq_len);

        for item in &items {
            inp_flat.extend(item.input.iter().map(|&id| id as i32));
            tgt_flat.extend(item.target.iter().map(|&id| id as i32));
        }

        let input = Tensor::<B, 1, Int>::from_data(
            TensorData::new(inp_flat, [batch_size * seq_len]),
            device,
        )
        .reshape([batch_size, seq_len]);

        let target = Tensor::<B, 1, Int>::from_data(
            TensorData::new(tgt_flat, [batch_size * seq_len]),
            device,
        )
        .reshape([batch_size, seq_len]);

        LmBatch { input, target }
    }
}
