/// Training — wired to the exact Burn 0.21 supervised-learning API.
///
/// Burn 0.21 API summary (what changed from older versions):
///
/// ┌─ Batcher ──────────────────────────────────────────────────────────────┐
/// │  trait Batcher<B: Backend, I, O> {                                      │
/// │      fn batch(&self, items: Vec<I>, device: &B::Device) -> O;           │
/// │  }   ← device injected, NOT stored in the struct                        │
/// └────────────────────────────────────────────────────────────────────────┘
///
/// ┌─ TrainStep (on Autodiff model) ────────────────────────────────────────┐
/// │  trait TrainStep {                                                       │
/// │      type Input:  Send + 'static;                                        │
/// │      type Output: ItemLazy + 'static;                                    │
/// │      fn step(&self, item: Self::Input) -> TrainOutput<Self::Output>;    │
/// │  }                                                                       │
/// └────────────────────────────────────────────────────────────────────────┘
///
/// ┌─ InferenceStep (on INNER model, i.e. M::InnerModule) ─────────────────┐
/// │  trait InferenceStep {                                                   │
/// │      type Input:  Send + 'static;                                        │
/// │      type Output: ItemLazy + 'static;                                    │
/// │      fn step(&self, item: Self::Input) -> Self::Output;                 │
/// │  }                                                                       │
/// └────────────────────────────────────────────────────────────────────────┘
///
/// ┌─ SupervisedTraining ───────────────────────────────────────────────────┐
/// │  SupervisedTraining::new(dir, dl_train, dl_val)                         │
/// │      where dl_train: Arc<dyn DataLoader<B,          M::Input>>          │
/// │            dl_valid: Arc<dyn DataLoader<B::Inner,   Inner::Input>>      │
/// │  .num_epochs(n).summary()                                               │
/// │  .launch(Learner::new(model, optim, lr))   → LearningResult             │
/// └────────────────────────────────────────────────────────────────────────┘
use burn::{
    config::Config,
    data::dataloader::DataLoaderBuilder,
    nn::loss::CrossEntropyLossConfig,
    optim::AdamConfig,
    prelude::*,
    record::{CompactRecorder, Recorder},
    tensor::backend::AutodiffBackend,
    train::{
        metric::LossMetric,
        ClassificationOutput, InferenceStep, Learner, SupervisedTraining, TrainOutput, TrainStep,
    },
};
use std::path::Path;
use std::sync::Arc;

use crate::{
    dataset::{LmBatch, LmBatcher, LmDataset},
    model::{HandcraftedTransformer, TransformerConfig},
    tokenizer::Tokenizer,
};

// ── Config ────────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TrainingConfig {
    pub model:     TransformerConfig,
    pub optimizer: AdamConfig,
    #[config(default = 30)]   pub num_epochs:    usize,
    #[config(default = 8)]    pub batch_size:    usize,
    #[config(default = 42)]   pub seed:          u64,
    #[config(default = 0.1)]  pub val_split:     f64,
    #[config(default = 32)]   pub seq_len:       usize,
    #[config(default = 1e-3)] pub learning_rate: f64,
}

// ── Shared forward ────────────────────────────────────────────────────────────

impl<B: Backend> HandcraftedTransformer<B> {
    fn lm_forward(&self, batch: LmBatch<B>) -> ClassificationOutput<B> {
        let [bs, sl] = batch.input.dims();
        let logits_3d = self.forward(batch.input);
        let vocab     = logits_3d.dims()[2];
        let logits    = logits_3d.reshape([bs * sl, vocab]);
        let targets   = batch.target.reshape([bs * sl]);
        let loss = CrossEntropyLossConfig::new()
            .init(&logits.device())
            .forward(logits.clone(), targets.clone());
        ClassificationOutput { loss, output: logits, targets }
    }
}

// ── TrainStep — implemented on HandcraftedTransformer<AutodiffBackend> ────────
//   Input  = LmBatch<B>              (batch from the TRAIN dataloader, B = Autodiff)
//   Output = ClassificationOutput<B>

impl<B: AutodiffBackend> TrainStep for HandcraftedTransformer<B> {
    type Input  = LmBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> TrainOutput<Self::Output> {
        let out   = self.lm_forward(item);
        let grads = out.loss.backward();
        TrainOutput::new(self, grads, out)
    }
}

// ── InferenceStep — implemented on M::InnerModule = HandcraftedTransformer<B::InnerBackend>
//   SupervisedTraining requires M::InnerModule: InferenceStep
//   Input  = LmBatch<B::InnerBackend>     (batch from the VAL dataloader)
//   Output = ClassificationOutput<B::InnerBackend>

impl<B: Backend> InferenceStep for HandcraftedTransformer<B> {
    type Input  = LmBatch<B>;
    type Output = ClassificationOutput<B>;

    fn step(&self, item: Self::Input) -> Self::Output {
        self.lm_forward(item)
    }
}

// ── run_training ──────────────────────────────────────────────────────────────

pub fn run_training<B: AutodiffBackend>(
    device: B::Device,
    corpus_path: &str,
    artifact_dir: &str,
) {
    println!("📖  Loading corpus from '{corpus_path}'…");
    let corpus = std::fs::read_to_string(corpus_path)
        .unwrap_or_else(|_| panic!("Cannot read '{corpus_path}'"));

    let tokenizer = Tokenizer::build_from_text(&corpus);
    println!("📚  Vocabulary size: {}", tokenizer.vocab_size());

    // ── Build config ──────────────────────────────────────────────────────
    let seq_len = 32usize;
    let model_cfg = TransformerConfig::new(tokenizer.vocab_size())
        .with_d_model(128)
        .with_n_heads(4)
        .with_n_layers(2)
        .with_d_ff(256)
        .with_max_seq_len(128)
        .with_dropout(0.1);

    let config = TrainingConfig::new(model_cfg, AdamConfig::new())
        .with_seq_len(seq_len)
        .with_num_epochs(30)
        .with_batch_size(8)
        .with_learning_rate(1e-3)
        .with_val_split(0.1);

    // ── Dataset split ─────────────────────────────────────────────────────
    let full    = LmDataset::new(&corpus, &tokenizer, config.seq_len);
    let n       = full.len();
    let n_val   = ((n as f64) * config.val_split) as usize;
    let n_train = n - n_val;
    println!("🗂   Train: {n_train}  Val: {n_val}");

    use burn::data::dataset::Dataset;
    let all: Vec<_> = (0..n).filter_map(|i| full.get(i)).collect();
    let train_ds = burn::data::dataset::InMemDataset::new(all[..n_train].to_vec());
    let val_ds   = burn::data::dataset::InMemDataset::new(all[n_train..].to_vec());

    // ── DataLoaders ───────────────────────────────────────────────────────
    // dl_train: DataLoader<B,               LmBatch<B>>               (Autodiff backend)
    // dl_valid: DataLoader<B::InnerBackend, LmBatch<B::InnerBackend>> (base backend)
    //
    // Both use the same stateless LmBatcher; the backend is injected via type inference.
    let dl_train: Arc<dyn burn::data::dataloader::DataLoader<B, LmBatch<B>>> =
        DataLoaderBuilder::new(LmBatcher)
            .batch_size(config.batch_size)
            .shuffle(config.seed)
            .num_workers(1)
            .build(train_ds);

    let dl_valid: Arc<dyn burn::data::dataloader::DataLoader<B::InnerBackend, LmBatch<B::InnerBackend>>> =
        DataLoaderBuilder::new(LmBatcher)
            .batch_size(config.batch_size)
            .num_workers(1)
            .build(val_ds);

    // ── Model ─────────────────────────────────────────────────────────────
    B::seed(&device, config.seed);
    let model = config.model.init::<B>(&device);
    println!("🔧  Parameters: {}", model.param_count());

    // ── Launch training ───────────────────────────────────────────────────
    // SupervisedTraining::new(dir, dl_train, dl_valid)
    // .launch(Learner::new(model, optim, lr))  → LearningResult { model, renderer }
    std::fs::create_dir_all(artifact_dir).ok();

    let training = SupervisedTraining::new(artifact_dir, dl_train, dl_valid)
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .num_epochs(config.num_epochs)
        .summary();

    let result  = training.launch(Learner::new(
        model,
        config.optimizer.init(),
        config.learning_rate,
    ));
    let trained = result.model;

    // ── Save artifacts ────────────────────────────────────────────────────
    CompactRecorder::new()
        .record(trained.into_record(), Path::new(artifact_dir).join("model"))
        .expect("Failed to save model");

    std::fs::write(
        Path::new(artifact_dir).join("words.json"),
        serde_json::to_string_pretty(&tokenizer.id_to_word).expect("json"),
    ).expect("Failed to write words.json");

    config.save(Path::new(artifact_dir).join("config.json"))
        .expect("Failed to save config");

    println!("\n✅  Done! → '{artifact_dir}'");
    println!("    Run `cargo run --release --bin generate`");
}
