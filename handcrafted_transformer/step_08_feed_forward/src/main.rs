/// Step 8: The Brain's Decision Filter (Feed-Forward & GELU)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   After our detectives gather their clues, the AI needs to process them and make decisions.
///   In our brain, we have neurons (brain cells). A neuron doesn't just pass everything along;
///   it decides if a signal is important enough to fire.
///
///   If we only had attention layers, our transformer would only be able to add and subtract
///   numbers (which is linear math). It wouldn't be able to learn complex logical rules.
///   The Feed-Forward Network (FFN) is where the "thinking" happens. It processes each word
///   separately. Inside it, we use a special math curve called GELU.
///   GELU acts like a dimmer switch:
///     - If a number is negative (unimportant noise), it squishes it down to zero.
///     - If a number is positive (important signal), it lets it pass through.

use burn::backend::NdArray;
use burn::module::Module;
use burn::nn::{
    Dropout, DropoutConfig,
    Linear, LinearConfig,
};
use burn::prelude::*;

type Backend = NdArray<f32>;

// ── Feed-Forward Network (Adapted from our main project model.rs) ───────────
#[derive(Module, Debug)]
pub struct FeedForward<B: burn::tensor::backend::Backend> {
    linear1: Linear<B>,
    linear2: Linear<B>,
    dropout: Dropout,
}

impl<B: burn::tensor::backend::Backend> FeedForward<B> {
    pub fn new(d_model: usize, d_ff: usize, dropout: f64, device: &B::Device) -> Self {
        FeedForward {
            linear1: LinearConfig::new(d_model, d_ff).init(device),
            linear2: LinearConfig::new(d_ff, d_model).init(device),
            dropout: DropoutConfig::new(dropout).init(),
        }
    }

    pub fn forward(&self, x: Tensor<B, 3>) -> Tensor<B, 3> {
        let x = self.linear1.forward(x);
        let x = burn::tensor::activation::gelu(x); // The GELU non-linear filter!
        let x = self.dropout.forward(x);
        self.linear2.forward(x)
    }
}

fn main() {
    println!("🧠 STEP 8: THE BRAIN'S DECISION FILTER (FEED-FORWARD)");
    println!("====================================================");
    println!("Welcome to Step 8! We will look at how the GELU filter works");
    println!("and run a mock coordinate tensor through a Feed-Forward layer.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // ── Step 8.1: Let's inspect how GELU filters numbers! ─────────────────────
    println!("📈 Step 8.1: Running numbers through the GELU activation function:");
    println!("   Input Value ➔ GELU Output Value");
    
    // We will test numbers from -3.0 to +3.0
    let test_inputs = vec![-3.0f32, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0];
    
    // Convert to a Burn tensor
    let input_tensor = Tensor::<Backend, 1>::from_data(
        TensorData::new(test_inputs.clone(), [test_inputs.len()]),
        &device
    );
    // Run GELU
    let gelu_outputs = burn::tensor::activation::gelu(input_tensor);
    let output_vec = gelu_outputs.into_data().to_vec::<f32>().unwrap();

    for (i, &inp) in test_inputs.iter().enumerate() {
        let out = output_vec[i];
        let status = if inp < 0.0 {
            "📴 Blocked/Squished"
        } else {
            "📶 Passed through"
        };
        println!("     {:>4.1}      ➔      {:>6.4}     ({})", inp, out, status);
    }
    println!();

    // ── Step 8.2: Run data through our FFN layer ─────────────────────────────
    let d_model = 4;
    let d_ff = 8;
    println!("🧱 Step 8.2: Initializing Feed-Forward Network layer:");
    println!("   - Input coordinate dimensions (d_model): {}", d_model);
    println!("   - Hidden brain cell size (d_ff):          {}", d_ff);
    println!();

    let ffn_layer = FeedForward::<Backend>::new(d_model, d_ff, 0.0, &device);

    // Mock input sequence of 3 words (shape [1 batch, 3 words, 4 coordinates])
    let mock_data = vec![
        0.5f32, -1.2f32, 2.0f32, -0.5f32,  // Word 1
        1.5f32, 0.8f32, -0.9f32, 0.1f32,   // Word 2
        -2.0f32, -0.5f32, 0.4f32, 1.1f32,  // Word 3
    ];
    let input = Tensor::<Backend, 3>::from_data(
        TensorData::new(mock_data, [1, 3, d_model]),
        &device
    );

    println!("   📥 Feed-Forward input shape:  {:?}", input.dims());
    let output = ffn_layer.forward(input);
    println!("   📤 Feed-Forward output shape: {:?}", output.dims());
    println!();

    println!("💡 Why do we need this?");
    println!("   Attention layers only shift vectors around (averaging coordinates).");
    println!("   The Feed-Forward layer acts as the individual processor for each word,");
    println!("   giving the model the power to say: 'If word A is next to word B,");
    println!("   AND it is a verb, then activate this meaning.'");
    println!();
    println!("🎉 Step 8 Complete! You now understand the decision-making filter of a transformer.");
}
