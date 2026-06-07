/// Text generation / inference — three decoding strategies.
use burn::{prelude::*, tensor::activation::softmax};

use crate::{model::HandcraftedTransformer, tokenizer::Tokenizer};

pub enum DecodeStrategy {
    Greedy,
    Sample { temperature: f32 },
    TopK { k: usize, temperature: f32 },
}

/// Generate `max_new_tokens` autoregressively, streaming each word to stdout.
pub fn generate<B: Backend>(
    model: &HandcraftedTransformer<B>,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_new_tokens: usize,
    strategy: &DecodeStrategy,
    max_seq_len: usize,
    device: &B::Device,
) -> String {
    let mut tokens: Vec<usize> = tokenizer.encode(prompt);
    if tokens.is_empty() { tokens.push(tokenizer.bos_id()); }

    for _ in 0..max_new_tokens {
        let context = if tokens.len() > max_seq_len {
            tokens[tokens.len() - max_seq_len..].to_vec()
        } else {
            tokens.clone()
        };
        let sl = context.len();

        let ids: Vec<i32> = context.iter().map(|&id| id as i32).collect();
        let input = Tensor::<B, 1, Int>::from_data(TensorData::new(ids, [sl]), device)
            .unsqueeze::<2>();

        let logits_3d = model.forward(input);
        let vocab = logits_3d.dims()[2];

        // Logits at last position → [vocab_size]
        let logits = logits_3d
            .slice([0..1, (sl - 1)..sl, 0..vocab])
            .reshape([vocab]);

        let next_id = pick_token::<B>(&logits, strategy);
        if next_id == tokenizer.eos_id() { break; }
        tokens.push(next_id);

        print!("{} ", tokenizer.decode(&[next_id]));
        use std::io::Write;
        std::io::stdout().flush().ok();
    }
    println!();
    tokenizer.decode(&tokens)
}

fn pick_token<B: Backend>(logits: &Tensor<B, 1>, strategy: &DecodeStrategy) -> usize {
    match strategy {
        DecodeStrategy::Greedy => {
            logits.clone().argmax(0).into_scalar().elem::<i64>() as usize
        }
        DecodeStrategy::Sample { temperature } => {
            let scaled = logits.clone() / (*temperature as f64);
            let probs  = softmax(scaled.unsqueeze::<2>(), 1).reshape([logits.dims()[0]]);
            multinomial(probs)
        }
        DecodeStrategy::TopK { k, temperature } => {
            let vocab = logits.dims()[0];
            let k = (*k).min(vocab);
            let (top_vals, top_idx) = topk(logits, k);
            let scaled = top_vals / (*temperature as f64);
            let probs  = softmax(scaled.unsqueeze::<2>(), 1).reshape([k]);
            top_idx[multinomial(probs)]
        }
    }
}

/// CDF-based multinomial sample.
fn multinomial<B: Backend>(probs: Tensor<B, 1>) -> usize {
    let p: Vec<f32> = probs.into_data().to_vec::<f32>().unwrap_or_default();
    let u: f32 = rand::random();
    let mut cum = 0.0f32;
    for (i, &pi) in p.iter().enumerate() {
        cum += pi;
        if cum >= u { return i; }
    }
    p.len().saturating_sub(1)
}

/// Return top-k (values, original indices) sorted by descending logit.
fn topk<B: Backend>(logits: &Tensor<B, 1>, k: usize) -> (Tensor<B, 1>, Vec<usize>) {
    let data: Vec<f32> = logits.clone().into_data().to_vec::<f32>().unwrap_or_default();
    let mut indexed: Vec<(usize, f32)> = data.into_iter().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    indexed.truncate(k);

    let indices: Vec<usize> = indexed.iter().map(|(i, _)| *i).collect();
    let values:  Vec<f32>   = indexed.iter().map(|(_, v)| *v).collect();

    let device = logits.device();
    let vals_t = Tensor::<B, 1>::from_data(TensorData::new(values, [k]), &device);
    (vals_t, indices)
}
