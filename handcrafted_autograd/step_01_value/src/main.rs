//! Step 1: A Number With a Slope (The `Value`)
//!
//! Run this step from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   A normal number like 3.0 is forgetful — it only knows its own size.
//!   Our `Value` is a number with a SECOND slot called `grad` (its "slope").
//!   `grad` answers the question: "if I wiggle this number up a tiny bit,
//!   how much does the final score change?"
//!
//!   Right now every slope is 0.0 — we haven't asked any questions yet.
//!   In Step 3 we'll write `backward()` to fill these slopes in automatically.
//!
//! Burn bridge:
//!   In `handcrafted_transformer`, every weight is a `Tensor` that secretly
//!   carries a gradient too. We're building the single-number version so you
//!   can see the gradient with your own eyes.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fmt;
use std::rc::Rc;

/// A scalar that carries BOTH a value (`data`) and its slope (`grad`).
///
/// Why `Rc<RefCell<..>>`? Soon many parts of a math expression will need to
/// SHARE and UPDATE the same number (e.g. one weight used in several places).
/// `Rc` = shared ownership, `RefCell` = "I can change what's inside."
/// Don't worry about the details yet — just read `Value` as "a smart number."
#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);

struct Inner {
    data: f64,        // the number itself, e.g. 3.0
    grad: f64,        // its slope — starts at 0.0, filled in by backprop later
    label: String,    // a human-friendly name, just for printing
}

impl Value {
    /// Make a fresh leaf value (an input or a weight).
    pub fn new(data: f64, label: &str) -> Value {
        Value(Rc::new(RefCell::new(Inner {
            data,
            grad: 0.0,
            label: label.to_string(),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn label(&self) -> String {
        self.0.borrow().label.clone()
    }

    /// Pretend backprop ran and gave this value a slope. (Just for the demo —
    /// the real, automatic version arrives in Step 3.)
    pub fn set_grad(&self, g: f64) {
        self.0.borrow_mut().grad = g;
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(
            f,
            "{:<10} Value(data={:+.4}, grad={:+.4})",
            self.label(),
            self.data(),
            self.grad()
        )
    }
}

fn main() {
    println!("🔢 STEP 1: A NUMBER WITH A SLOPE (THE `Value`)");
    println!("=============================================");
    println!("A `Value` is a number that ALSO remembers its slope (`grad`).");
    println!();

    // 1. Make a few leaf values — think of these as inputs and weights.
    let x = Value::new(3.0, "x");
    let w = Value::new(-1.5, "w");
    let b = Value::new(0.5, "b");

    println!("📦 Three fresh values (slopes all start at 0.0):");
    println!("   {x}");
    println!("   {w}");
    println!("   {b}");
    println!();

    // 2. The `data` is the number you set. The `grad` is empty for now.
    println!("🔍 Peeking inside x:");
    println!("   x.data() = {:.1}   ← the number itself", x.data());
    println!("   x.grad() = {:.1}   ← its slope (nobody has asked a question yet)", x.grad());
    println!();

    // 3. Imagine backprop discovered "wiggling x up by 1 raises the score by 2".
    //    That fact is a slope of 2.0. We'll compute this AUTOMATICALLY in Step 3;
    //    here we just set it by hand to see what the slot is for.
    x.set_grad(2.0);
    println!("✍️  Pretend backprop told us x has slope 2.0:");
    println!("   {x}");
    println!("   Meaning: nudging x from 3.0 → 4.0 would raise the final score by ~2.0.");
    println!();

    println!("💡 Why two slots?");
    println!("   • data  → tells the model what to OUTPUT.");
    println!("   • grad  → tells the model how to IMPROVE.");
    println!("   Training is just: read every grad, then nudge every data.");
    println!();
    println!("🎉 Step 1 complete! Next: teach + and × to remember their parents.");
}
