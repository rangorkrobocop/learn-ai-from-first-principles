/// Step 5: Focusing Your Flashlight (Single-Head Attention)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   When you see the word "bank" in the sentence "I sat by the river bank",
///   how do you know it's not a money bank? You look at "river"!
///   Attention is like shining a flashlight on other words in the sentence to get context.
///
///   There are three parts to this:
///     1. Query (Q): The word asking: "What am I looking for?" (e.g., "bank" is looking for context).
///     2. Key (K): The words advertising themselves: "Here is what I am about!" (e.g., "river" says "I'm about water").
///     3. Value (V): The actual content/meaning we extract once we find a match.
///
/// Middle School Math Connection:
///   1. Dot Product: To find out how well a Query matches a Key, we multiply their coordinate numbers
///      together and add them up.
///        Matching Score = (Q_x * K_x) + (Q_y * K_y)
///      If the Query and Key point in similar directions, their dot product is a high positive number!
///   2. Softmax: We convert these matching scores into percentages (weights) that add up to 100%.
///      If we have scores [2.0, 1.0, -1.0], softmax turns them into roughly [70%, 26%, 4%].

use burn::backend::NdArray;
use burn::prelude::*;
use burn::tensor::activation::softmax;

type Backend = NdArray<f32>;

fn main() {
    println!("🔦 STEP 5: FOCUSING YOUR FLASHLIGHT (ATTENTION)");
    println!("==============================================");
    println!("Welcome to Step 5! We will simulate how a word ('bank') shines its");
    println!("attention flashlight on other words in a sentence to understand its context.");
    println!();

    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // 1. Let's define the words in our sentence:
    // "the", "river", "bank", "robber"
    let words = vec!["the", "river", "bank", "robber"];
    let seq_len = words.len();
    let d_k = 2; // Size of our key/query vectors: 2D coordinates

    println!("📝 Sentence: \"the river bank robber\"");
    println!("🔦 We want the word \"bank\" (at index 2) to look at all the words.");
    println!();

    // 2. Mock Query vector for "bank" (asking for water or money context)
    // Let's say it is looking for water-related context: [1.2, 0.5]
    let query_data = vec![1.2f32, 0.5f32];
    let query = Tensor::<Backend, 1>::from_data(TensorData::new(query_data.clone(), [d_k]), &device)
        .unsqueeze::<2>(); // reshape to [1, d_k]

    // 3. Mock Key vectors for all 4 words:
    // "the":    [0.1, 0.1] (boring word)
    // "river":  [1.0, 0.8] (strong water coordinate - matches our query!)
    // "bank":   [0.5, 0.2] (self-match)
    // "robber": [-0.8, -0.5] (money/crime coordinate - opposite of our water query!)
    let keys_data = vec![
        0.1f32,  0.1f32,  // "the"
        1.0f32,  0.8f32,  // "river"
        0.5f32,  0.2f32,  // "bank"
        -0.8f32, -0.5f32, // "robber"
    ];
    let keys = Tensor::<Backend, 2>::from_data(TensorData::new(keys_data, [seq_len, d_k]), &device);

    println!("🔑 Step 5.1: Keys for each word (what they represent):");
    println!("  * \"the\"    ➔ [{:.2}, {:.2}]", 0.1, 0.1);
    println!("  * \"river\"  ➔ [{:.2}, {:.2}] (water-aligned)", 1.0, 0.8);
    println!("  * \"bank\"   ➔ [{:.2}, {:.2}]", 0.5, 0.2);
    println!("  * \"robber\" ➔ [{:.2}, {:.2}] (money-aligned)", -0.8, -0.5);
    println!();

    // 4. Calculate matching scores: Q @ K^T / sqrt(d_k)
    // Dot product: query [1, d_k] multiplied by keys [seq_len, d_k]^T = [1, seq_len]
    let keys_t = keys.swap_dims(0, 1); // transpose keys to [d_k, seq_len]
    let scale = (d_k as f64).sqrt();
    let scores = query.matmul(keys_t) / scale; // [1, seq_len]
    
    let scores_vec = scores.clone().into_data().to_vec::<f32>().unwrap();

    println!("📐 Step 5.2: Calculating Matching Scores (Dot Products / sqrt(dim)):");
    for i in 0..seq_len {
        println!("  * \"bank\" looking at \"{:8}\" ➔ Raw score: {:.4}", words[i], scores_vec[i]);
    }
    println!();

    // 5. Apply Softmax to get percentages (weights that sum to 100%)
    let weights = softmax(scores, 1); // dim 1 is the sequence dimension
    let weights_vec = weights.into_data().to_vec::<f32>().unwrap();

    println!("📊 Step 5.3: Converting to Attention Weights (Softmax Percentages):");
    for i in 0..seq_len {
        let percentage = weights_vec[i] * 100.0;
        
        // Draw a simple bar chart to represent the flashlight brightness!
        let num_chars = (percentage / 5.0).round() as usize;
        let bar = "█".repeat(num_chars) + &"░".repeat(20 - num_chars);

        println!("  * \"bank\" ➔ \"{:8}\" : {:5.1}%  [{}]", words[i], percentage, bar);
    }
    println!();

    println!("💡 What happened?");
    println!("   The flashlight is shining brightest (most percentage weight) on \"river\"!");
    println!("   This is because the Query coordinates for \"bank\" matched the Key coordinates");
    println!("   for \"river\" very closely, giving it the highest dot product. The word");
    println!("   \"bank\" has successfully focused on \"river\" to get its water context!");
    println!();
    println!("🎉 Step 5 Complete! You now know the core math of Transformer Attention.");
}
