/// ============================================================
/// Handcrafted Transformer — built component by component
/// ============================================================
///
/// Architecture (decoder-only, GPT style):
///
///   token_ids  ──► Embedding ──► + PosEncoding
///                                      │
///                              ┌───────▼───────┐
///                              │  N × Block    │   each block:
///                              │  ┌──────────┐ │     LayerNorm
///                              │  │  MH Attn │ │     MultiHeadSelfAttention + residual
///                              │  │  FFN     │ │     LayerNorm
///                              │  └──────────┘ │     FeedForward + residual
///                              └───────┬───────┘
///                                      │
///                                 LayerNorm
///                                      │
///                                  Linear ──► logits [vocab_size]
///
/// Every operation here is explicitly written out so you can follow the
/// maths from "Attention Is All You Need" step by step.
///
/// BURN NOTE: Tensors stored directly inside a #[derive(Module)] struct
/// are treated as non-parameter buffers (they move with .to_device() but
/// are not returned by .parameters()). That is exactly what we want for
/// the sinusoidal position table.

use burn::{
    config::Config,
    module::Module,
    nn::{
        Dropout, DropoutConfig,
        Embedding, EmbeddingConfig,
        LayerNorm, LayerNormConfig,
        Linear, LinearConfig,
    },
    prelude::*,
    tensor::activation::softmax,
};

// ────────────────────────────────────────────────────────────────────────────
// Configuration
// ────────────────────────────────────────────────────────────────────────────

#[derive(Config, Debug)]
pub struct TransformerConfig {
    /// Number of unique tokens (vocabulary size).
    pub vocab_size: usize,
    /// Dimensionality of every embedding / hidden state.
    #[config(default = 128)]
    pub d_model: usize,
    /// Number of attention heads. Must divide d_model evenly.
    #[config(default = 4)]
    pub n_heads: usize,
    /// Number of transformer blocks stacked on top of each other.
    #[config(default = 2)]
    pub n_layers: usize,
    /// Hidden dim of the position-wise feed-forward network.
    #[config(default = 256)]
    pub d_ff: usize,
    /// Maximum sequence length (for the sinusoidal position table).
    #[config(default = 128)]
    pub max_seq_len: usize,
    /// Dropout probability applied in attention and FFN.
    #[config(default = 0.1)]
    pub dropout: f64,
}

impl TransformerConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> HandcraftedTransformer<B> {
        assert!(
            self.d_model % self.n_heads == 0,
            "d_model ({}) must be divisible by n_heads ({})",
            self.d_model,
            self.n_heads
        );

        let token_embedding = EmbeddingConfig::new(self.vocab_size, self.d_model).init(device);

        // Pre-compute sinusoidal positional encodings.
        // Stored as a Tensor inside the Module — Burn treats this as a
        // non-differentiable buffer: it moves with the model but is not
        // included in the parameter list.
        let pos_encoding = build_sinusoidal_encoding(self.max_seq_len, self.d_model, device);

        let blocks = (0..self.n_layers)
            .map(|_| {
                TransformerBlock::new(
                    self.d_model,
                    self.n_heads,
                    self.d_ff,
                    self.dropout,
                    device,
                )
            })
            .collect();

        let final_norm = LayerNormConfig::new(self.d_model).init(device);
        let lm_head = LinearConfig::new(self.d_model, self.vocab_size)
            .with_bias(false)
            .init(device);

        HandcraftedTransformer {
            token_embedding,
            pos_encoding,
            blocks,
            final_norm,
            lm_head,
            d_model: self.d_model,
            n_heads: self.n_heads,
        }
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Sinusoidal positional encoding  (no learned parameters)
//
//   PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
//   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
//
// Shape: [max_seq_len, d_model]
//
// WHY: Attention has no built-in notion of order. Without positional
// encoding, "the cat sat" and "sat the cat" look identical to the model.
// We inject position information by adding a unique vector to each position.
// Sinusoids allow the model to extrapolate to unseen lengths and let it
// learn relative positions (sin(a+b) = linear combination of sin(a), cos(a)).
// ────────────────────────────────────────────────────────────────────────────

fn build_sinusoidal_encoding<B: Backend>(
    max_seq_len: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data = vec![0f32; max_seq_len * d_model];

    for pos in 0..max_seq_len {
        for i in 0..(d_model / 2) {
            // Frequency decreases as i increases → low-i dims change fast
            // (fine-grained position), high-i dims change slowly (coarse).
            let angle = pos as f32 / f32::powf(10_000.0, (2 * i) as f32 / d_model as f32);
            data[pos * d_model + 2 * i]     = angle.sin(); // even dims
            data[pos * d_model + 2 * i + 1] = angle.cos(); // odd dims
        }
    }

    Tensor::<B, 1>::from_data(
        TensorData::new(data, [max_seq_len * d_model]),
        device,
    )
    .reshape([max_seq_len, d_model])
}

// ────────────────────────────────────────────────────────────────────────────
// Causal (autoregressive) attention mask
//
// Returns [seq_len, seq_len] with −1e9 in every upper-triangle cell.
// Adding this to attention scores before softmax makes exp(−1e9) ≈ 0,
// so future tokens receive zero attention weight.
//
// Example for seq_len = 4:
//   [  0,  -∞,  -∞,  -∞ ]   token 0 can only see itself
//   [  0,   0,  -∞,  -∞ ]   token 1 can see 0 and 1
//   [  0,   0,   0,  -∞ ]
//   [  0,   0,   0,   0 ]   token 3 can see all
// ────────────────────────────────────────────────────────────────────────────

fn causal_mask<B: Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in 0..seq_len {
            if col > row {
                data[row * seq_len + col] = -1e9;
            }
        }
    }
    Tensor::<B, 1>::from_data(
        TensorData::new(data, [seq_len * seq_len]),
        device,
    )
    .reshape([seq_len, seq_len])
}

// ────────────────────────────────────────────────────────────────────────────
// Multi-Head Self-Attention
//
// The core idea: instead of one big attention computation, split into
// h independent "heads", each working in a lower-dimensional subspace.
// Each head can specialise in a different kind of relationship:
//   head 0 → syntactic agreement
//   head 1 → coreference
//   head 2 → positional proximity  ... etc.
//
// Steps:
//  1. Project X → Q, K, V  via learned weight matrices (all heads at once)
//  2. Reshape: split the d_model dimension into [n_heads, d_k]
//  3. For each head: scores = Q @ K^T / sqrt(d_k)
//  4. Add causal mask, then softmax → attention weights
//  5. output = weights @ V
//  6. Concat heads, project through W_o
// ────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct MultiHeadSelfAttention<B: Backend> {
    /// Fused Q projection for all heads: [d_model → d_model]
    w_q: Linear<B>,
    /// Fused K projection for all heads: [d_model → d_model]
    w_k: Linear<B>,
    /// Fused V projection for all heads: [d_model → d_model]
    w_v: Linear<B>,
    /// Output projection (merges heads): [d_model → d_model]
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

    /// x:    [batch, seq_len, d_model]
    /// mask: [seq_len, seq_len]   additive causal mask
    /// → out [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        let [batch, seq_len, _d] = x.dims();
        let d_k = self.d_model / self.n_heads; // per-head key/query dimension

        // ── 1. Project input to Q, K, V (all heads simultaneously) ──────
        // Each result: [batch, seq_len, d_model]
        let q = self.w_q.forward(x.clone());
        let k = self.w_k.forward(x.clone());
        let v = self.w_v.forward(x);

        // ── 2. Split heads ───────────────────────────────────────────────
        // [batch, seq_len, d_model]
        //   → reshape [batch, seq_len, n_heads, d_k]
        //   → swap    [batch, n_heads, seq_len, d_k]
        let split = |t: Tensor<B, 3>| {
            t.reshape([batch, seq_len, self.n_heads, d_k])
                .swap_dims(1, 2)
        };
        let q = split(q); // [batch, n_heads, seq_len, d_k]
        let k = split(k);
        let v = split(v);

        // ── 3. Scaled dot-product attention ─────────────────────────────
        // scores = Q @ K^T / sqrt(d_k)
        //   Q:   [batch, n_heads, seq_len, d_k]
        //   K^T: [batch, n_heads, d_k,     seq_len]
        //   → scores: [batch, n_heads, seq_len, seq_len]
        //
        // The 1/sqrt(d_k) scaling prevents dot-products from growing large
        // as d_k increases, which would push softmax into saturation
        // (vanishing gradients).
        let k_t = k.swap_dims(2, 3);
        let scale = (d_k as f64).sqrt();
        let scores = q.matmul(k_t) / scale;

        // ── 4. Add causal mask ───────────────────────────────────────────
        // Broadcast [seq_len, seq_len] → [batch, n_heads, seq_len, seq_len]
        let mask_4d = mask.clone()
            .unsqueeze::<3>()   // [1, seq_len, seq_len]
            .unsqueeze::<4>();  // [1, 1, seq_len, seq_len]
        let scores = scores + mask_4d.expand([batch, self.n_heads, seq_len, seq_len]);

        // ── 5. Softmax over key dimension → attention weights ─────────────
        let weights = softmax(scores, 3); // dim 3 = key axis
        let weights = self.dropout.forward(weights);

        // ── 6. Weighted sum of values ─────────────────────────────────────
        // [batch, n_heads, seq_len, seq_len] @ [batch, n_heads, seq_len, d_k]
        // = [batch, n_heads, seq_len, d_k]
        let context = weights.matmul(v);

        // ── 7. Merge heads and project output ─────────────────────────────
        // [batch, n_heads, seq_len, d_k]
        //   → swap [batch, seq_len, n_heads, d_k]
        //   → reshape [batch, seq_len, d_model]   (concat heads)
        let context = context
            .swap_dims(1, 2)
            .reshape([batch, seq_len, self.d_model]);

        // Final linear projection mixes information across heads.
        self.w_o.forward(context)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Position-wise Feed-Forward Network
//
//   FFN(x) = GELU( x @ W1 + b1 ) @ W2 + b2
//
// Applied to each position independently (same weights for all positions).
// d_ff is typically 4× d_model; here we use 2× to keep the model tiny.
//
// WHY GELU: smoother than ReLU, widely used in GPT-family models.
// The non-linearity is where the model gains expressive power beyond
// what pure attention (which is linear in V) can represent.
// ────────────────────────────────────────────────────────────────────────────

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

    /// x: [batch, seq_len, d_model] → [batch, seq_len, d_model]
    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x);            // GELU non-linearity
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Transformer Block  (Pre-LN variant)
//
// Original paper (Post-LN):  x = LayerNorm( x + SubLayer(x) )
// Pre-LN (what we use):      x = x + SubLayer( LayerNorm(x) )
//
// Pre-LN trains more stably without learning-rate warmup and is standard
// in modern implementations (GPT-2, GPT-3, LLaMA all use it).
// ────────────────────────────────────────────────────────────────────────────

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

    /// x: [batch, seq_len, d_model], mask: [seq_len, seq_len]
    pub fn forward(&self, x: Tensor<B, 3>, mask: &Tensor<B, 2>) -> Tensor<B, 3> {
        // Sub-layer 1: masked self-attention
        //   Normalise first, then attend, then add residual.
        //   The residual connection lets gradients flow backwards
        //   without vanishing through many layers.
        let attn_out = self.attn.forward(self.norm1.forward(x.clone()), mask);
        let x = x + attn_out;

        // Sub-layer 2: position-wise feed-forward
        let ffn_out = self.ffn.forward(self.norm2.forward(x.clone()));
        x + ffn_out
    }
}

// ────────────────────────────────────────────────────────────────────────────
// Full Model
// ────────────────────────────────────────────────────────────────────────────

#[derive(Module, Debug)]
pub struct HandcraftedTransformer<B: Backend> {
    pub token_embedding: Embedding<B>,
    /// Sinusoidal table [max_seq_len, d_model].
    /// Stored as a Tensor — not a learned parameter, but moves with the model.
    pub pos_encoding: Tensor<B, 2>,
    pub blocks: Vec<TransformerBlock<B>>,
    pub final_norm: LayerNorm<B>,
    /// Language-model head: projects hidden → vocab logits.
    pub lm_head: Linear<B>,
    pub d_model: usize,
    pub n_heads: usize,
}

impl<B: Backend> HandcraftedTransformer<B> {
    /// Forward pass.
    ///
    /// `input_ids`: [batch, seq_len] — integer token IDs
    /// Returns:     [batch, seq_len, vocab_size] — unnormalised logits
    pub fn forward(&self, input_ids: Tensor<B, 2, Int>) -> Tensor<B, 3> {
        let [batch, seq_len] = input_ids.dims();

        // ── Token embeddings ─────────────────────────────────────────────
        // Each integer ID is looked up in a [vocab_size, d_model] table.
        // We scale by √d_model so that the initial embedding magnitude
        // matches the positional encoding magnitude (both ~O(1)).
        let scale = (self.d_model as f64).sqrt() as f32;
        let tok_emb = self.token_embedding.forward(input_ids) * scale;
        // tok_emb: [batch, seq_len, d_model]

        // ── Positional encodings ─────────────────────────────────────────
        // Slice the precomputed table to the actual sequence length,
        // then broadcast over the batch dimension.
        let pos_emb = self
            .pos_encoding
            .clone()
            .slice([0..seq_len, 0..self.d_model])  // [seq_len, d_model]
            .unsqueeze::<3>()                         // [1, seq_len, d_model]
            .expand([batch, seq_len, self.d_model]);

        // ── Sum embeddings ────────────────────────────────────────────────
        let mut x = tok_emb + pos_emb; // [batch, seq_len, d_model]

        // ── Build causal mask (reused for all blocks) ────────────────────
        let mask = causal_mask::<B>(seq_len, &x.device());

        // ── Stack of transformer blocks ───────────────────────────────────
        for block in &self.blocks {
            x = block.forward(x, &mask);
        }

        // ── Final layer norm → logits ─────────────────────────────────────
        self.lm_head.forward(self.final_norm.forward(x))
        // [batch, seq_len, vocab_size]
    }

    /// Count total trainable parameters (uses Burn's built-in counter).
    pub fn param_count(&self) -> usize {
        self.num_params()
    }
}
