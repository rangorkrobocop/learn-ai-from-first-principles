/// Step 7: A Team of Detectives (Multi-Head Attention)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   If you only have one flashlight (one attention head), you can only focus on
///   one word relationship at a time. But human language is complex!
///   So, we build a team of detectives (called heads).
///     - Detective 1: Looks for verbs (Who did what?).
///     - Detective 2: Looks for pronouns (Who is "he" or "she"?).
///     - Detective 3: Looks for descriptions (What color/size is it?).
///
///   Each detective works independently in their own room (their own coordinate subspace).
///   Then, they all meet, combine their notes (concatenate their outputs), and write a
///   final master report (output projection).

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig,
    Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

type Backend = NdArray<f32>;

// ── Multi-Head Self-Attention (Adapted from our main project model.rs) ───────
#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    w_q: Linear<B>,
    w_k: Linear<B>,
    w_v: Linear<B>,
    w_o: Linear<B>,
    dropout: Dropout,
    n_heads: usize,
    d_model: usize,
}

impl<B: Backend> MultiHeadSelfAttention<B> {
    pub fn new(d_model: usize, n_heads: usize, dropout: f64, device: &B::Device) -> Self {
        let linear = |d_in, d_out| LinearConfig::new(d_in, d_out).with_bias(false).init(device);
        MultiHeadSelfAttention {
            w_q: linear(d_model, d_model),
            w_k: linear(d_model, d_model),
            w_v: linear(d_model, d_model),
            w_o: linear(d_model, d_model),
            dropout: DropoutConfig::new(dropout).init(),
            n_heads,
            d_model,
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, seq_len, _d] = x.dims();
        let d_k = self.d_model / self.n_heads;

        // 1. Project Q, K, V
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        println!("   🔍 1. Input projected into Q, K, V. Shape: {:?}", q.dims());

        // 2. Split heads
        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, seq_len, self.n_heads, d_k])
                .swap_dims(1, 2)
        };
        let q = split(q); // [batch, n_heads, seq_len, d_k]
        let k = split(k);
        let v = split(v);

        println!("   👥 2. Split into {} heads! Subspace shape: {:?}", self.n_heads, q.dims());

        // 3. Attention scores: Q @ K^T / sqrt(d_k)
        let k_t = k.swap_dims(2, 3);
        let scale = (d_k as f64).sqrt();
        let scores = q.matmul(k_t) / scale;

        // 4. Add causal mask
        let mask_4d = mask.clone()
            .unsqueeze::<3>()
            .unsqueeze::<4>();
        let scores = scores + mask_4d.expand([batch, self.n_heads, seq_len, seq_len]);

        // 5. Softmax
        let weights = softmax(scores, 3);
        let weights = self.dropout.forward(weights);

        // 6. Weighted sum of values
        let context = weights.matmul(v);
        println!("   🔦 3. Attention computed on each head. Shape: {:?}", context.dims());

        // 7. Concat heads and project out
        let context = context
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.d_model]);
        println!("   🤝 4. Concatenated heads back together. Shape: {:?}", context.dims());

        let out = self.w_o.forward(context);
        println!("   📝 5. Output projected. Final block output shape: {:?}", out.dims());
        out
    }
}

/// Helper function to create causal mask (same as Step 6)
fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in 0..seq_len {
            if col > row { data[row * seq_len + col] = -1e9; }
        }
    }
    Tensor::<B, 1>::from_data(TensorData::new(data, [seq_len * seq_len]), device)
        .reshape([seq_len, seq_len])
}

fn main() {
    println!("👥 STEP 7: A TEAM OF DETECTIVES (MULTI-HEAD ATTENTION)");
    println!("=====================================================");
    println!("Welcome to Step 7! We will run Multi-Head Self-Attention");
    println!("and watch the tensor shape change as we split it for our detectives.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // ── Setup parameters ──────────────────────────────────────────────────────
    let seq_len = 3;
    let d_model = 4; // Word coordinates have 4 numbers
    let n_heads = 2; // We have 2 detectives
    println!("📍 Word coordinate size (d_model): {}", d_model);
    println!("👥 Number of detectives (n_heads): {}", n_heads);
    println!("📍 Coordinate size per head (d_k): {}", d_model / n_heads);
    println!();

    // Initialize our multi-head attention block
    let attention_layer = MultiHeadSelfAttention::<Backend>::new(d_model, n_heads, 0.0, &device);

    // Create a mock input sequence of 3 words (shape [1 batch, 3 words, 4 coordinates])
    let mock_input_data = vec![
        1.0f32, 0.5f32, -0.2f32, 0.8f32, // Word 1
        -0.5f32, 1.2f32, 0.3f32, -0.4f32, // Word 2
        0.2f32, -0.1f32, 1.0f32, 0.5f32,  // Word 3
    ];
    let input = Tensor::<Backend, 3>::from_data(
        TensorData::new(mock_input_data, [1, seq_len, d_model]),
        &device
    );
    let mask = causal_mask::<Backend>(seq_len, &device);

    println!("🏃‍♂️ Step 7.1: Running Multi-Head Attention forward pass...");
    println!("   📥 Input tensor shape: {:?}", input.dims());
    println!();

    let output = attention_layer.forward(input, &mask);
    println!();

    println!("💡 What did we see?");
    println!("   1. The input starts with shape [1, 3, 4] (1 sentence, 3 words, 4 dimensions).");
    println!("   2. We split the 4 dimensions into 2 heads. Each head gets shape [1, 2, 3, 2]");
    println!("      (1 sentence, 2 heads, 3 words, 2 dimensions).");
    println!("   3. Each head does attention on its own 2 dimensions.");
    println!("   4. We paste (concatenate) the 2-dim outputs back into 4 dimensions: [1, 3, 4].");
    println!("   5. A final projection mixes information between the two heads.");
    println!();
    println!("🎉 Step 7 Complete! You understand how teams of attention heads split and solve language problems.");
}
