/// Train the handcrafted transformer on `data/corpus.txt`.
///
/// Usage:
///   cargo run --bin train
///   cargo run --release --bin train   # much faster
///
/// Trained artifacts are saved to `./artifacts/`.
use burn::backend::{Autodiff, NdArray};

type Backend = NdArray<f32>;
type AutodiffBackend = Autodiff<Backend>;

fn main() {
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    step_10_complete_gpt::training::run_training::<AutodiffBackend>(
        device,
        "data/corpus.txt",
        "artifacts",
    );
}
