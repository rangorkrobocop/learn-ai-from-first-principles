/// Step 3: Words as GPS Coordinates on a Map (Embeddings)
///
/// Run this step by running in your terminal:
///   cargo run
///
/// Analogy:
///   If you just assign random number IDs to words, the computer doesn't know what
///   they mean. It doesn't know that "cat" and "dog" are similar (both pets), or that
///   "computer" is different.
///   An Embedding turns each word ID into GPS coordinates (like X and Y) on a 2D map.
///   On this map, we want similar words to be placed close to each other:
///     - "cat" might be at (1.2, 0.9)
///     - "dog" might be at (1.3, 0.8)  ➔ Very close!
///     - "computer" might be at (-5.0, 4.2)  ➔ Far away!
///
///   By looking at how close coordinates are, the computer learns what words mean!
///
/// Basic Mathematics Connection:
///   To find the distance between two points on a map, we use the Pythagorean theorem!
///   Distance = SquareRoot( (X2 - X1)² + (Y2 - Y1)² )

use burn::backend::NdArray;
use burn::nn::{EmbeddingConfig, Embedding};
use burn::prelude::*;

type Backend = NdArray<f32>;

fn main() {
    println!("📍 STEP 3: WORDS AS GPS COORDINATES (EMBEDDINGS)");
    println!("===============================================");
    println!("Welcome to Step 3! We will use the Burn framework to turn word IDs");
    println!("into 2D map coordinates (X, Y) and calculate their semantic distance.");
    println!();

    // 1. Setup backend and device
    let device = burn::backend::ndarray::NdArrayDevice::Cpu;

    // 2. We have a tiny vocabulary of 5 words:
    // ID 0: "<pad>"
    // ID 1: "<unk>"
    // ID 2: "cat"
    // ID 3: "dog"
    // ID 4: "computer"
    let vocab_size = 5;
    let d_model = 2; // 2D coordinates: [X, Y]

    println!("📚 Vocabulary Size: {} words", vocab_size);
    println!("📍 Dimensions per word (d_model): {} (meaning an X and a Y coordinate)", d_model);
    println!();

    // 3. Initialize an embedding layer.
    // When we start, these coordinates are completely random!
    let embedding: Embedding<Backend> = EmbeddingConfig::new(vocab_size, d_model).init(&device);

    // 4. Look up coordinates for "cat" (ID 2), "dog" (ID 3), and "computer" (ID 4)
    // We create a tensor of IDs: [2, 3, 4] and feed it to the embedding layer.
    let token_ids = Tensor::<Backend, 1, Int>::from_data(
        TensorData::new(vec![2, 3, 4], [3]), 
        &device
    ).unsqueeze::<2>(); // reshape to [1, 3] (1 sentence, 3 words)

    let coordinates = embedding.forward(token_ids);
    
    // Extract the raw coordinates to print them nicely
    let coords_data = coordinates.into_data().to_vec::<f32>().unwrap();
    
    let cat_x = coords_data[0];
    let cat_y = coords_data[1];
    let dog_x = coords_data[2];
    let dog_y = coords_data[3];
    let comp_x = coords_data[4];
    let comp_y = coords_data[5];

    println!("📖 Step 3.1: Word Coordinates (Randomly Initialized):");
    println!("  🐱 \"cat\" (ID 2)       ➔ Coordinates: (X: {:.4}, Y: {:.4})", cat_x, cat_y);
    println!("  🐶 \"dog\" (ID 3)       ➔ Coordinates: (X: {:.4}, Y: {:.4})", dog_x, dog_y);
    println!("  💻 \"computer\" (ID 4)  ➔ Coordinates: (X: {:.4}, Y: {:.4})", comp_x, comp_y);
    println!();

    // 5. Calculate Pythagorean distances!
    let dist_cat_dog = ((dog_x - cat_x).powi(2) + (dog_y - cat_y).powi(2)).sqrt();
    let dist_cat_comp = ((comp_x - cat_x).powi(2) + (comp_y - cat_y).powi(2)).sqrt();

    println!("📐 Step 3.2: Calculating Distances using the Pythagorean Theorem:");
    println!("  Distance = SquareRoot( (X₂ - X₁)² + (Y₂ - Y₁)² )");
    println!();
    println!("  🐾 Distance between \"cat\" and \"dog\":      {:.4}", dist_cat_dog);
    println!("  🔌 Distance between \"cat\" and \"computer\": {:.4}", dist_cat_comp);
    println!();

    println!("💡 What does this mean?");
    println!("   Right now, the distances are random because the coordinates are random.");
    println!("   During training, the Coach (Optimizer) checks if we get questions wrong,");
    println!("   and adjusts the coordinates. It will slide 'cat' and 'dog' closer");
    println!("   together because they share contexts, and push 'computer' away!");
    println!();
    println!("🎉 Step 3 Complete! You understand how words get map positions.");
}
