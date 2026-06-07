/// Generate text with the trained transformer.
///
/// Usage:
///   cargo run --release --bin generate
///
/// Requires artifacts/ to exist (run `cargo run --release --bin train` first).
use burn::{
    backend::NdArray,
    prelude::*,
    record::{CompactRecorder, Recorder},
};
use step_10_complete_gpt::{
    generation::{generate, DecodeStrategy},
    tokenizer::Tokenizer,
    training::TrainingConfig,
};

type Backend = NdArray<f32>;

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // ── Load config saved during training ─────────────────────────────────
    let config = TrainingConfig::load("artifacts/config.json").unwrap_or_else(|_| {
        eprintln!("❌  Could not load artifacts/config.json");
        eprintln!("    Run `cargo run --release --bin train` first.");
        std::process::exit(1);
    });

    // ── Rebuild tokenizer from saved word list ────────────────────────────
    let words_json = std::fs::read_to_string("artifacts/words.json")
        .expect("Could not read artifacts/words.json");
    let id_to_word: Vec<String> =
        serde_json::from_str(&words_json).expect("Malformed words.json");
    let word_to_id = id_to_word
        .iter()
        .enumerate()
        .map(|(i, w)| (w.clone(), i))
        .collect();
    let tokenizer = Tokenizer { word_to_id, id_to_word };

    // ── Load trained model weights ────────────────────────────────────────
    let model = config.model.init::<Backend>(&device);
    let record = CompactRecorder::new()
        .load("artifacts/model".into(), &device)
        .expect("Could not load artifacts/model — did training complete?");
    let model = model.load_record(record);

    let max_seq = config.model.max_seq_len;
    println!("🤖  Model loaded ({} params).  Generating…\n", model.param_count());

    // ── Greedy ────────────────────────────────────────────────────────────
    println!("━━━  Greedy decoding  ━━━");
    print!("→ ");
    generate(&model, &tokenizer, "the transformer", 50,
             &DecodeStrategy::Greedy, max_seq, &device);

    // ── Temperature sampling ──────────────────────────────────────────────
    println!("━━━  Temperature sampling  T=0.8  ━━━");
    print!("→ ");
    generate(&model, &tokenizer, "attention is", 50,
             &DecodeStrategy::Sample { temperature: 0.8 }, max_seq, &device);

    // ── Top-5 sampling ────────────────────────────────────────────────────
    println!("━━━  Top-5 sampling  T=1.0  ━━━");
    print!("→ ");
    generate(&model, &tokenizer, "the model learns", 50,
             &DecodeStrategy::TopK { k: 5, temperature: 1.0 }, max_seq, &device);

    println!("\n💡  Tip: change the prompts or strategy in src/bin/generate.rs");
}
