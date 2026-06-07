/// Step 9: The Factory Assembly Line (The Transformer Block)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   In a car factory, you don't build a car all in one spot. The car moves on a
///   conveyor belt (conveyor line) through different stations.
///
///   A Transformer Block is one station in our AI factory:
///     1. Wash the parts (LayerNorm): Standardizes coordinates so they don't get too big.
///     2. Inspect context (Multi-Head Attention): Look at neighboring words.
///     3. Keep the blueprint (Residual Connection): We add the attention changes
///        back to the original input. This ensures we never lose track of the original words.
///     4. Wash the parts again (LayerNorm).
///     5. Think and filter (Feed-Forward & GELU): Process the features.
///     6. Keep the blueprint again (Residual Connection): Add the FFN changes back.
///
///   By stacking multiple blocks, the AI gets smarter at each station!

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig,
    LayerNorm, LayerNormConfig,
    Linear, LinearConfig,
};
use burn::prelude::*;
use burn::tensor::activation::softmax;

type Backend = NdArray<f32>;

// ── Multi-Head Self-Attention ────────────────────────────────────────────────
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
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, seq_len, self.n_heads, d_k])
                .swap_dims(1, 2)
        };
        let q = split(q);
        let k = split(k);
        let v = split(v);

        let k_t = k.swap_dims(2, 3);
        let scale = (d_k as f64).sqrt();
        let scores = q.matmul(k_t) / scale;

        let mask_4d = mask.clone().unsqueeze::<3>().unsqueeze::<4>();
        let scores = scores + mask_4d.expand([batch, self.n_heads, seq_len, seq_len]);

        let weights = softmax(scores, 3);
        let weights = self.dropout.forward(weights);

        let context = weights.matmul(v);
        let context = context.swap_dims(1, 2).reshape([batch, seq_len, self.d_model]);
        self.w_o.forward(context)
    }
}

// ── Feed-Forward Network ─────────────────────────────────────────────────────
#[derive(Module, Debug)]
pub struct FeedForward<B: Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: Backend> FeedForward<B> {
    pub fn new(d_model: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        FeedForward {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x);
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

// ── Transformer Block (Pre-LN variant) ───────────────────────────────────────
#[derive(Module, Debug)]
pub struct TransformerBlock<B: Backend> {
    norm1: LayerNorm<B>,
    attn: MultiHeadSelfAttention<B>,
    norm2: LayerNorm<B>,
    ffn: FeedForward<B>,
}

impl<B: Backend> TransformerBlock<B> {
    pub fn new(
        d_model: usize,
        n_heads: usize,
        d_ff: usize,
        dropout: f64,
        device: &B::Device,
    ) -> Self {
        TransformerBlock {
            norm1: LayerNormConfig::new(d_model).init(device),
            attn: MultiHeadSelfAttention::new(d_model, n_heads, dropout, device),
            norm2: LayerNormConfig::new(d_model).init(device),
            ffn: FeedForward::new(d_model, d_ff, dropout, device),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        // 1. LayerNorm ➔ Attention ➔ Add back to input (Residual Connection)
        let normalized_x = self.norm1.forward(x.clone());
        let attn_out = self.attn.forward(normalized_x, mask);
        let x = x + attn_out; // Residual addition!

        // 2. LayerNorm ➔ Feed-Forward ➔ Add back to input (Residual Connection)
        let normalized_x2 = self.norm2.forward(x.clone());
        let ffn_out = self.ffn.forward(normalized_x2);
        x + ffn_out // Residual addition!
    }
}

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
    println!("🏗️ STEP 9: THE CONVEYOR STATION (TRANSFORMER BLOCK)");
    println!("==================================================");
    println!("Welcome to Step 9! We will assemble all the individual components");
    println!("into a single Transformer Block and watch the data flow through it.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Setup dimensions
    let seq_len = 3;
    let d_model = 4;
    let n_heads = 2;
    let d_ff = 8;

    println!("🧱 Block Configuration:");
    println!("   - Coordinate Dimensions (d_model): {}", d_model);
    println!("   - Attention heads:                  {}", n_heads);
    println!("   - Feed-Forward Brain cells (d_ff):  {}", d_ff);
    println!();

    // Initialize the complete Transformer Block
    let block = TransformerBlock::<Backend>::new(d_model, n_heads, d_ff, 0.1, &device);

    // Mock input sequence of 3 words (shape [1 batch, 3 words, 4 coordinates])
    let mock_data = vec![
        1.2f32, -0.3f32, 0.5f32, 0.8f32,  // Word 1
        -0.8f32, 1.5f32, 0.2f32, -0.4f32, // Word 2
        0.1f32, -0.9f32, 1.1f32, 0.3f32,  // Word 3
    ];
    let input = Tensor::<Backend, 3>::from_data(
        TensorData::new(mock_data, [1, seq_len, d_model]),
        &device
    );
    let mask = causal_mask::<Backend>(seq_len, &device);

    println!("🏃‍♂️ Step 9.1: Processing our sentence through the Transformer Block...");
    println!("   📥 Input shape: {:?}", input.dims());
    println!();

    let output = block.forward(input, &mask);
    println!();
    
    println!("   📤 Block output shape: {:?}", output.dims());
    println!();

    println!("💡 Why do we use Residual Connections (x = x + sub_layer(x))?");
    println!("   Imagine writing a long game of telephone. By the time you reach the 12th player,");
    println!("   the original message is completely distorted. Residual connections carry the");
    println!("   original word meanings alongside the changes. This allows us to stack");
    println!("   dozens of blocks without losing the original words!");
    println!();
    println!("🎉 Step 9 Complete! You have successfully built the core factory block of a Transformer.");
}
