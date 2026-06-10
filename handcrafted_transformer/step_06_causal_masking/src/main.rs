/// Step 6: No Cheating! (Causal Masking)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   Imagine taking a fill-in-the-blank spelling test. If the question is:
///     "The robot ___ on the computer"
///   You have to guess "sat" using ONLY the words before the blank: "The robot".
///   If you could look to the right, you would see "on the computer", which is cheating!
///
///   When we train our AI, we feed it a whole sentence at once (for speed).
///   To stop it from cheating, we use a Causal Mask (a screen). It acts like placing
///   your hand over the right side of the page and sliding it one word to the right
///   at a time.
///
/// Basic Mathematics Connection:
///   We block future words by adding a HUGE negative number (like negative one billion: -1,000,000,000)
///   to their attention scores. When we do the percentage split (softmax), any score that is
///   negative infinity gets EXACTLY 0% of the attention.

use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::activation::softmax;

type Backend = NdArray<f32>;

/// Create an additive causal mask of shape [seq_len, seq_len].
/// It contains 0.0 on and below the diagonal, and -1e9 (negative infinity) above it.
fn causal_mask<B: burn::tensor::backend::Backend>(seq_len: usize, device: &B::Device) -> Tensor<B, 2> {
    let mut data = vec![0f32; seq_len * seq_len];
    for row in 0..seq_len {
        for col in 0..seq_len {
            if col > row {
                // Future token! Block it!
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

fn main() {
    println!("🚫 STEP 6: NO CHEATING! (CAUSAL MASKING)");
    println!("=======================================");
    println!("Welcome to Step 6! We will build a triangular mask to block the future");
    println!("and see how the math forces future attention weights to exactly 0%.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;
    let seq_len = 4;
    let words = vec!["the", "robot", "sat", "on"];

    println!("📝 Sentence words: {:?}", words);
    println!();

    // 1. Create a causal mask
    let mask = causal_mask::<Backend>(seq_len, &device);
    let mask_data = mask.clone().into_data().to_vec::<f32>().unwrap();

    println!("🛡️  Step 6.1: The Causal Mask Matrix [seq_len, seq_len]:");
    for row in 0..seq_len {
        print!("   Row {} (Word: {:<5}) ➔ [ ", row, words[row]);
        for col in 0..seq_len {
            let val = mask_data[row * seq_len + col];
            if val < -100.0 {
                print!("-∞, "); // Represent negative infinity
            } else {
                print!("{:.1}, ", val);
            }
        }
        println!("]");
    }
    println!();

    // 2. Imagine we have some raw attention matching scores (before masking)
    // Let's create a [seq_len, seq_len] matrix with some mock scores.
    let raw_scores_data = vec![
        1.5f32, 2.0f32, 0.5f32, 3.0f32, // Row 0 ("the" looking at: the, robot, sat, on)
        0.8f32, 1.2f32, 2.5f32, 0.2f32, // Row 1 ("robot" looking at: the, robot, sat, on)
        2.0f32, 0.5f32, 1.5f32, 1.0f32, // Row 2 ("sat" looking at: the, robot, sat, on)
        1.0f32, 1.0f32, 1.0f32, 1.0f32, // Row 3 ("on" looking at: the, robot, sat, on)
    ];
    let raw_scores = Tensor::<Backend, 2>::from_data(
        TensorData::new(raw_scores_data.clone(), [seq_len, seq_len]),
        &device
    );

    println!("📝 Step 6.2: Raw Attention Scores (Before Masking - Cheating!):");
    for row in 0..seq_len {
        print!("   \"{:5}\" looking at ➔ [ ", words[row]);
        for col in 0..seq_len {
            print!("{:.1}, ", raw_scores_data[row * seq_len + col]);
        }
        println!("]");
    }
    println!();

    // 3. Add the mask to the scores!
    // Adding -1e9 makes future scores super negative.
    let masked_scores = raw_scores + mask;
    let masked_data = masked_scores.clone().into_data().to_vec::<f32>().unwrap();

    println!("🛠️  Step 6.3: Masked Scores (Future scores set to -Infinity):");
    for row in 0..seq_len {
        print!("   \"{:5}\" looking at ➔ [ ", words[row]);
        for col in 0..seq_len {
            let val = masked_data[row * seq_len + col];
            if val < -100.0 {
                print!("-∞, ");
            } else {
                print!("{:.1}, ", val);
            }
        }
        println!("]");
    }
    println!();

    // 4. Run through softmax to get final attention weights
    let weights = softmax(masked_scores, 1);
    let weights_data = weights.into_data().to_vec::<f32>().unwrap();

    println!("📊 Step 6.4: Final Attention Weights (Softmax Percentages):");
    for row in 0..seq_len {
        print!("   \"{:5}\" looks at ➔ [ ", words[row]);
        for col in 0..seq_len {
            let percentage = weights_data[row * seq_len + col] * 100.0;
            print!("{:5.1}%, ", percentage);
        }
        println!("]");
    }
    println!();

    println!("💡 What do you notice?");
    println!("   Look at Row 1 (\"robot\"): it pays 40.1% attention to \"the\", 59.9% to \"robot\",");
    println!("   and exactly 0.0% to \"sat\" and \"on\"! It has no idea the future words even exist.");
    println!("   This triangular pattern is called the Causal Mask!");
    println!();
    println!("🎉 Step 6 Complete! You now understand how we prevent the model from cheating.");
}
