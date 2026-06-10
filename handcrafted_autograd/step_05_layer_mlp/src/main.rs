//! Step 5: Stacking Cells Into a Brain (Layers & the MLP)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   One neuron is weak. A LAYER is a row of neurons that all see the same
//!   inputs. An MLP (Multi-Layer Perceptron) stacks several layers so the
//!   output of one becomes the input of the next — a brain made of brain cells.
//!
//!   We hide all of this behind a `Module` TRAIT (Rust's word for "interface").
//!   That is the exact mechanism Burn's `#[derive(Module)]` gives the transformer.
//!
//! (ENGINE block is the same scalar autograd from Step 3. NN block is new.)

#![allow(dead_code)]

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

// ============================================================================
//  THE ENGINE  (scalar autograd — identical to Step 3)
// ============================================================================

#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);
struct Inner {
    data: f64,
    grad: f64,
    op: &'static str,
    prev: Vec<Value>,
    backward: Option<Box<dyn Fn(f64)>>,
}
impl Value {
    fn wrap(inner: Inner) -> Value {
        Value(Rc::new(RefCell::new(inner)))
    }
    pub fn new(data: f64) -> Value {
        Value::wrap(Inner { data, grad: 0.0, op: "leaf", prev: vec![], backward: None })
    }
    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }
    pub fn grad(&self) -> f64 {
        self.0.borrow().grad
    }
    pub fn set_data(&self, x: f64) {
        self.0.borrow_mut().data = x;
    }
    pub fn zero_grad(&self) {
        self.0.borrow_mut().grad = 0.0;
    }
    fn id(&self) -> *const RefCell<Inner> {
        Rc::as_ptr(&self.0)
    }
    pub fn tanh(&self) -> Value {
        let t = self.0.borrow().data.tanh();
        let out = Value::wrap(Inner { data: t, grad: 0.0, op: "tanh", prev: vec![self.clone()], backward: None });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| { a.0.borrow_mut().grad += (1.0 - t * t) * g; }));
        out
    }
    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let y = if x > 0.0 { x } else { 0.0 };
        let out = Value::wrap(Inner { data: y, grad: 0.0, op: "relu", prev: vec![self.clone()], backward: None });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| { a.0.borrow_mut().grad += if x > 0.0 { g } else { 0.0 }; }));
        out
    }
    pub fn powf(&self, n: f64) -> Value {
        let base = self.0.borrow().data;
        let out = Value::wrap(Inner { data: base.powf(n), grad: 0.0, op: "pow", prev: vec![self.clone()], backward: None });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| { a.0.borrow_mut().grad += n * base.powf(n - 1.0) * g; }));
        out
    }
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<Inner>> = HashSet::new();
        fn build(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<*const RefCell<Inner>>) {
            if visited.insert(v.id()) {
                for parent in v.0.borrow().prev.iter() { build(parent, topo, visited); }
                topo.push(v.clone());
            }
        }
        build(self, &mut topo, &mut visited);
        self.0.borrow_mut().grad = 1.0;
        for node in topo.iter().rev() {
            let g = node.0.borrow().grad;
            let inner = node.0.borrow();
            if let Some(bw) = inner.backward.as_ref() { bw(g); }
        }
    }
}
impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::wrap(Inner { data: self.data() + rhs.data(), grad: 0.0, op: "+", prev: vec![self.clone(), rhs.clone()], backward: None });
        let (a, b) = (self.clone(), rhs.clone());
        out.0.borrow_mut().backward = Some(Box::new(move |g| { a.0.borrow_mut().grad += g; b.0.borrow_mut().grad += g; }));
        out
    }
}
impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        let out = Value::wrap(Inner { data: self.data() * rhs.data(), grad: 0.0, op: "*", prev: vec![self.clone(), rhs.clone()], backward: None });
        let (a, b) = (self.clone(), rhs.clone());
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            let (ad, bd) = (a.data(), b.data());
            a.0.borrow_mut().grad += bd * g;
            b.0.borrow_mut().grad += ad * g;
        }));
        out
    }
}
impl Neg for &Value {
    type Output = Value;
    fn neg(self) -> Value { self * &Value::new(-1.0) }
}
impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value { self + &(-rhs) }
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value(data={:+.4}, grad={:+.4})", self.data(), self.grad())
    }
}

// ============================================================================
//  THE NN LIBRARY  —  Module trait, Linear layer, MLP (Burn-shaped)
// ============================================================================

/// The shared interface every learnable block implements.
/// (This is the scalar twin of Burn's `Module` trait.)
pub trait Module {
    fn forward(&self, x: &[Value]) -> Vec<Value>;
    fn parameters(&self) -> Vec<Value>;
}

/// Builder for a `Linear` layer — mirrors Burn's `LinearConfig::new(a, b).init(..)`.
pub struct LinearConfig {
    d_in: usize,
    d_out: usize,
    nonlin: bool,
}
impl LinearConfig {
    pub fn new(d_in: usize, d_out: usize) -> Self {
        Self { d_in, d_out, nonlin: false }
    }
    /// If true, apply a tanh squash after the linear part (a hidden layer).
    pub fn with_nonlin(mut self, yes: bool) -> Self {
        self.nonlin = yes;
        self
    }
    /// Create the layer with random weights in [-1, 1] and zero biases.
    pub fn init(&self, rng: &mut StdRng) -> Linear {
        let weight = (0..self.d_out)
            .map(|_| (0..self.d_in).map(|_| Value::new(rng.gen_range(-1.0..1.0))).collect())
            .collect();
        let bias = (0..self.d_out).map(|_| Value::new(0.0)).collect();
        Linear { weight, bias, nonlin: self.nonlin }
    }
}

/// A fully-connected layer: a whole row of neurons sharing the inputs.
pub struct Linear {
    weight: Vec<Vec<Value>>, // [d_out][d_in]
    bias: Vec<Value>,        // [d_out]
    nonlin: bool,
}
impl Module for Linear {
    fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.weight
            .iter()
            .zip(self.bias.iter())
            .map(|(row, b)| {
                let mut act = b.clone();
                for (w, xi) in row.iter().zip(x.iter()) {
                    act = &act + &(w * xi);
                }
                if self.nonlin { act.tanh() } else { act }
            })
            .collect()
    }
    fn parameters(&self) -> Vec<Value> {
        let mut p = vec![];
        for row in &self.weight {
            for w in row {
                p.push(w.clone());
            }
        }
        for b in &self.bias {
            p.push(b.clone());
        }
        p
    }
}

/// A Multi-Layer Perceptron: several `Linear` layers stacked.
/// Hidden layers use tanh; the final layer is left linear (raw scores out).
pub struct Mlp {
    layers: Vec<Linear>,
}
impl Mlp {
    pub fn new(n_in: usize, layer_sizes: &[usize], rng: &mut StdRng) -> Self {
        let mut sizes = vec![n_in];
        sizes.extend_from_slice(layer_sizes);
        let n = layer_sizes.len();
        let layers = (0..n)
            .map(|i| {
                let last = i == n - 1;
                LinearConfig::new(sizes[i], sizes[i + 1]).with_nonlin(!last).init(rng)
            })
            .collect();
        Mlp { layers }
    }
}
impl Module for Mlp {
    fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers {
            out = layer.forward(&out);
        }
        out
    }
    fn parameters(&self) -> Vec<Value> {
        self.layers.iter().flat_map(|l| l.parameters()).collect()
    }
}

// ============================================================================
//  DEMO
// ============================================================================

fn main() {
    println!("🧱 STEP 5: STACKING CELLS INTO A BRAIN (MLP)");
    println!("============================================");
    println!("An MLP = Linear layers stacked. We hide it behind a `Module` trait —");
    println!("the same interface Burn's #[derive(Module)] gives the transformer.");
    println!();

    // A fixed seed → the same "random" brain every run (great for tutorials).
    let mut rng = StdRng::seed_from_u64(42);

    // 3 inputs → hidden layer of 4 → hidden layer of 4 → 1 output.
    let model = Mlp::new(3, &[4, 4, 1], &mut rng);
    println!("🧠 Architecture: 3 inputs → [4, 4, 1]");
    println!("   Tunable parameters: {}", model.parameters().len());
    println!();

    // One forward pass.
    let x = [Value::new(2.0), Value::new(3.0), Value::new(-1.0)];
    let y = model.forward(&x);
    println!("🧮 forward([2.0, 3.0, -1.0]):");
    println!("   output = {:+.4}", y[0].data());
    println!();

    // Backprop still works, untouched, through the whole stack.
    y[0].backward();
    let first = &model.parameters()[0];
    println!("🌊 backward() flows through every layer. Example knob:");
    println!("   first weight: data={:+.4}, grad={:+.4}", first.data(), first.grad());
    println!();

    println!("💡 The brain runs, but its output is random — it has never been");
    println!("   told what 'right' looks like. Next: a scorecard (loss).");
    println!();
    println!("🎉 Step 5 complete! We have a trainable model. Time to grade it.");
}
