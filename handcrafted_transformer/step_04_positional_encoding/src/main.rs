/// Step 4: Adding Page Numbers (Positional Encoding)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   Imagine reading a recipe where the words are completely scrambled. Order matters!
///   In "Attention" (which we will build in Step 5), there is no concept of word order.
///   To a basic attention searchlight, "the dog bit the boy" and "the boy bit the dog"
///   look exactly the same because they contain the same words.
///
///   To solve this, we add "Page Numbers" or "Position Stickers" to our word GPS coordinates.
///   We use smooth mathematical waves (sines and cosines) to generate a unique coordinate pattern 
///   for Position 0, Position 1, Position 2, etc. Then, we simply ADD this position coordinate
///   directly to our word coordinate!
///
/// Middle School Math Connection:
///   Sine (sin) and Cosine (cos) are mathematical functions that describe wave patterns,
///   like a pendulum swinging or a wave in the ocean. The output of sine and cosine is always
///   a number between -1.0 and 1.0. By using waves of different speeds (frequencies), we get
///   a unique wave value combination for every position!

use burn::backend::NdArray;
use burn::prelude::*;

type Backend = NdArray<f32>;

/// Build sinusoidal positional encodings.
///
/// PE(pos, 2i)   = sin(pos / 10000^(2i/d_model))
/// PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
fn build_sinusoidal_encoding<B: burn::tensor::backend::Backend>(
    max_seq_len: usize,
    d_model: usize,
    device: &B::Device,
) -> Tensor<B, 2> {
    let mut data = vec![0f32; max_seq_len * d_model];

    for pos in 0..max_seq_len {
        for i in 0..(d_model / 2) {
            // Speed of the wave decreases as i increases.
            // Even dimensions get sine waves, odd dimensions get cosine waves.
            let angle = pos as f32 / f32::powf(10_000.0, (2 * i) as f32 / d_model as f32);
            data[pos * d_model + 2 * i]     = angle.sin(); // Even coordinate index
            data[pos * d_model + 2 * i + 1] = angle.cos(); // Odd coordinate index
        }
    }

    Tensor::<B, 1>::from_data(
        TensorData::new(data, [max_seq_len * d_model]),
        device,
    )
    .reshape([max_seq_len, d_model])
}

fn main() {
    println!("📖 STEP 4: ADDING PAGE NUMBERS (POSITIONAL ENCODING)");
    println!("===================================================");
    println!("Welcome to Step 4! We will create wave-based coordinates to represent");
    println!("the positions of words in a sentence and add them to our word embeddings.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // Let's use 4 dimensions for our coordinates to make it easy to read
    let d_model = 4; 
    let max_seq_len = 5;

    println!("📍 Coordinate Dimensions (d_model): {}", d_model);
    println!("📏 Maximum Sequence Length:          {}", max_seq_len);
    println!();

    // 1. Build the positional encoding table
    let pos_encoding = build_sinusoidal_encoding::<Backend>(max_seq_len, d_model, &device);
    let pos_data = pos_encoding.clone().into_data().to_vec::<f32>().unwrap();

    println!("🌊 Step 4.1: Inspecting the Position Stickers (Sinusoidal Encodings):");
    for pos in 0..3 {
        let idx = pos * d_model;
        println!(
            "  Position {} Sticker ➔ Coordinates: [{:.4}, {:.4}, {:.4}, {:.4}]",
            pos, pos_data[idx], pos_data[idx + 1], pos_data[idx + 2], pos_data[idx + 3]
        );
    }
    println!();

    // 2. Let's see what happens to a word's coordinates when placed at different positions
    // Imagine the word "cat" has a base embedding coordinate of [1.0, 1.0, 1.0, 1.0]
    let cat_embedding = vec![1.0, 1.0, 1.0, 1.0];
    println!("🐱 Base word coordinate for \"cat\": {:?}", cat_embedding);
    println!();

    println!("📝 Step 4.2: Placing \"cat\" at different positions in a sentence:");

    // Cat at Position 0
    let cat_at_0: Vec<f32> = cat_embedding.iter().enumerate().map(|(i, &val)| val + pos_data[0 * d_model + i]).collect();
    println!("  * \"cat\" as word #1 (Position 0):");
    println!("    Base ({:?}) + Position 0 Sticker ({:?})", cat_embedding, &pos_data[0..d_model]);
    println!("    ➔ New Coordinates: {:.4?}", cat_at_0);
    println!();

    // Cat at Position 2
    let cat_at_2: Vec<f32> = cat_embedding.iter().enumerate().map(|(i, &val)| val + pos_data[2 * d_model + i]).collect();
    println!("  * \"cat\" as word #3 (Position 2):");
    println!("    Base ({:?}) + Position 2 Sticker ({:?})", cat_embedding, &pos_data[2 * d_model..3 * d_model]);
    println!("    ➔ New Coordinates: {:.4?}", cat_at_2);
    println!();

    println!("💡 Why do this?");
    println!("   By adding the position waves to the word coordinates, the resulting coordinate");
    println!("   is unique for both the word's meaning AND its place in the sentence. The AI");
    println!("   can now tell if \"cat\" is at the start, middle, or end of a sentence!");
    println!();
    println!("🎉 Step 4 Complete! You now understand how transformers keep track of word order.");
}
