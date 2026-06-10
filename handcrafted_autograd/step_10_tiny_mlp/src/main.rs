//! Step 10: Train a Real Net (The Tiny MLP)  —  GRADUATION
//!
//! Run from inside this folder with:
//!   cargo run --release        (release is ~10x faster for this one)
//!
//! Analogy:
//!   We make a 2D dataset — dots INSIDE a circle (class +1) vs OUTSIDE (class -1) —
//!   and train an MLP to tell them apart. A circle is not a straight line, so the
//!   network must bend its decision boundary. At the end we DRAW that boundary in
//!   ASCII so you can literally see what the network learned.
//!
//!   Every line below — the autograd, the layers, the loss, Adam, the loop — is
//!   something you built in Steps 1–9. Nothing is imported magic.
//!
//! Burn bridge:
//!   Swap `Value` for `Tensor`, this 2-input net for a 12-million-parameter GPT,
//!   and the circle for "predict the next word," and you have
//!   handcrafted_transformer. Same engine. Bigger numbers.

#![allow(dead_code)]

use rand::{rngs::StdRng, Rng, SeedableRng};
use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

// ============================================================================
//  THE ENGINE  (identical to Step 3)
// ============================================================================

#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);
struct Inner { data: f64, grad: f64, op: &'static str, prev: Vec<Value>, backward: Option<Box<dyn Fn(f64)>> }
impl Value {
    fn wrap(inner: Inner) -> Value { Value(Rc::new(RefCell::new(inner))) }
    pub fn new(data: f64) -> Value { Value::wrap(Inner { data, grad: 0.0, op: "leaf", prev: vec![], backward: None }) }
    pub fn data(&self) -> f64 { self.0.borrow().data }
    pub fn grad(&self) -> f64 { self.0.borrow().grad }
    pub fn set_data(&self, x: f64) { self.0.borrow_mut().data = x; }
    pub fn zero_grad(&self) { self.0.borrow_mut().grad = 0.0; }
    fn id(&self) -> *const RefCell<Inner> { Rc::as_ptr(&self.0) }
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
impl Neg for &Value { type Output = Value; fn neg(self) -> Value { self * &Value::new(-1.0) } }
impl Sub for &Value { type Output = Value; fn sub(self, rhs: &Value) -> Value { self + &(-rhs) } }
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { write!(f, "Value(data={:+.4}, grad={:+.4})", self.data(), self.grad()) }
}

// ============================================================================
//  THE NN LIBRARY  (Module / Linear / Mlp — identical to Step 5)
// ============================================================================

pub trait Module {
    fn forward(&self, x: &[Value]) -> Vec<Value>;
    fn parameters(&self) -> Vec<Value>;
}
pub struct LinearConfig { d_in: usize, d_out: usize, nonlin: bool }
impl LinearConfig {
    pub fn new(d_in: usize, d_out: usize) -> Self { Self { d_in, d_out, nonlin: false } }
    pub fn with_nonlin(mut self, yes: bool) -> Self { self.nonlin = yes; self }
    pub fn init(&self, rng: &mut StdRng) -> Linear {
        let weight = (0..self.d_out).map(|_| (0..self.d_in).map(|_| Value::new(rng.gen_range(-1.0..1.0))).collect()).collect();
        let bias = (0..self.d_out).map(|_| Value::new(0.0)).collect();
        Linear { weight, bias, nonlin: self.nonlin }
    }
}
pub struct Linear { weight: Vec<Vec<Value>>, bias: Vec<Value>, nonlin: bool }
impl Module for Linear {
    fn forward(&self, x: &[Value]) -> Vec<Value> {
        self.weight.iter().zip(self.bias.iter()).map(|(row, b)| {
            let mut act = b.clone();
            for (w, xi) in row.iter().zip(x.iter()) { act = &act + &(w * xi); }
            if self.nonlin { act.tanh() } else { act }
        }).collect()
    }
    fn parameters(&self) -> Vec<Value> {
        let mut p = vec![];
        for row in &self.weight { for w in row { p.push(w.clone()); } }
        for b in &self.bias { p.push(b.clone()); }
        p
    }
}
pub struct Mlp { layers: Vec<Linear> }
impl Mlp {
    pub fn new(n_in: usize, layer_sizes: &[usize], rng: &mut StdRng) -> Self {
        let mut sizes = vec![n_in];
        sizes.extend_from_slice(layer_sizes);
        let n = layer_sizes.len();
        let layers = (0..n).map(|i| {
            let last = i == n - 1;
            LinearConfig::new(sizes[i], sizes[i + 1]).with_nonlin(!last).init(rng)
        }).collect();
        Mlp { layers }
    }
}
impl Module for Mlp {
    fn forward(&self, x: &[Value]) -> Vec<Value> {
        let mut out = x.to_vec();
        for layer in &self.layers { out = layer.forward(&out); }
        out
    }
    fn parameters(&self) -> Vec<Value> { self.layers.iter().flat_map(|l| l.parameters()).collect() }
}

// ============================================================================
//  THE LOSS + OPTIMIZER  (MSE + Adam — identical to Steps 6 & 8)
// ============================================================================

pub fn mse(preds: &[Value], targets: &[f64]) -> Value {
    let mut loss = Value::new(0.0);
    for (p, t) in preds.iter().zip(targets.iter()) {
        let diff = p - &Value::new(*t);
        loss = &loss + &diff.powf(2.0);
    }
    &loss * &Value::new(1.0 / preds.len() as f64)
}

pub struct Adam { lr: f64, beta1: f64, beta2: f64, eps: f64, t: i32, m: Vec<f64>, v: Vec<f64> }
impl Adam {
    pub fn new(lr: f64) -> Self { Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: vec![], v: vec![] } }
    pub fn zero_grad(&self, params: &[Value]) { for p in params { p.zero_grad(); } }
    pub fn step(&mut self, params: &[Value]) {
        if self.m.len() != params.len() { self.m = vec![0.0; params.len()]; self.v = vec![0.0; params.len()]; }
        self.t += 1;
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t));
            p.set_data(p.data() - self.lr * m_hat / (v_hat.sqrt() + self.eps));
        }
    }
}

// ============================================================================
//  DATA + HELPERS
// ============================================================================

/// One labeled 2D point: position (x, y) and class (+1.0 inside, -1.0 outside).
struct Point {
    x: f64,
    y: f64,
    label: f64,
}

/// True boundary: inside a circle of radius √2.5 → class +1, else -1.
fn true_label(x: f64, y: f64) -> f64 {
    if x * x + y * y < 2.5 { 1.0 } else { -1.0 }
}

/// Run the model on a raw (x, y) and return its single output score.
fn predict(model: &Mlp, x: f64, y: f64) -> f64 {
    let input = [Value::new(x), Value::new(y)];
    model.forward(&input)[0].data()
}

/// Fraction of points the model classifies correctly (by sign of the score).
fn accuracy(model: &Mlp, data: &[Point]) -> f64 {
    let correct = data
        .iter()
        .filter(|p| predict(model, p.x, p.y).signum() == p.label.signum())
        .count();
    correct as f64 / data.len() as f64
}

// ============================================================================
//  DEMO  —  build data, train, draw the learned boundary
// ============================================================================

fn main() {
    println!("🏁 STEP 10: TRAIN A REAL NET (THE TINY MLP) — GRADUATION");
    println!("=======================================================");
    println!("Task: learn to tell dots INSIDE a circle (+1) from OUTSIDE (-1).");
    println!();

    // ── 1. Make a dataset of 100 random labeled points in [-2, 2]² ──
    let mut data_rng = StdRng::seed_from_u64(1);
    let n = 100;
    let data: Vec<Point> = (0..n)
        .map(|_| {
            let x = data_rng.gen_range(-2.0..2.0);
            let y = data_rng.gen_range(-2.0..2.0);
            Point { x, y, label: true_label(x, y) }
        })
        .collect();
    let pos = data.iter().filter(|p| p.label > 0.0).count();
    println!("📊 Dataset: {n} points  ({pos} inside / {} outside the circle)", n - pos);
    println!();

    // ── 2. Build the model: 2 inputs → [16, 16, 1] ──
    let mut init_rng = StdRng::seed_from_u64(42);
    let model = Mlp::new(2, &[16, 16, 1], &mut init_rng);
    let mut adam = Adam::new(0.05);
    println!("🧠 Model: 2 → [16, 16, 1]   ({} parameters)", model.parameters().len());
    println!();

    // Pre-compute the fixed targets once.
    let targets: Vec<f64> = data.iter().map(|p| p.label).collect();

    // ── 3. Train (full-batch gradient descent with Adam) ──
    let epochs = 100;
    println!("🔁 Training {epochs} epochs:");
    for epoch in 0..epochs {
        let preds: Vec<Value> = data
            .iter()
            .map(|p| {
                let input = [Value::new(p.x), Value::new(p.y)];
                model.forward(&input)[0].clone()
            })
            .collect();
        let loss = mse(&preds, &targets);

        let params = model.parameters();
        adam.zero_grad(&params);
        loss.backward();
        adam.step(&params);

        if epoch % 20 == 0 || epoch == epochs - 1 {
            println!(
                "   epoch {:>3}  loss = {:.5}  accuracy = {:.1}%",
                epoch,
                loss.data(),
                accuracy(&model, &data) * 100.0
            );
        }
    }
    println!();

    // ── 4. Draw the learned decision boundary ──
    println!("🗺️  Learned decision boundary  ('#' = model says inside, '·' = outside):");
    println!("    (the dotted ring is the TRUE circle the model was trying to find)");
    println!();
    let rows = 21;
    let cols = 43;
    for r in 0..rows {
        // y goes from +2 (top) down to -2 (bottom)
        let y = 2.0 - 4.0 * (r as f64) / (rows as f64 - 1.0);
        let mut line = String::from("    ");
        for c in 0..cols {
            let x = -2.0 + 4.0 * (c as f64) / (cols as f64 - 1.0);
            let said_inside = predict(&model, x, y) > 0.0;
            // Is this cell near the TRUE circle edge? (radius² ≈ 2.5)
            let r2 = x * x + y * y;
            let on_true_edge = (r2 - 2.5).abs() < 0.25;
            let ch = if on_true_edge {
                'o' // mark the real boundary for comparison
            } else if said_inside {
                '#'
            } else {
                '·'
            };
            line.push(ch);
        }
        println!("{line}");
    }
    println!();

    println!("💡 The '#' region should hug the 'o' ring — the network bent a straight-line");
    println!("   tool into a circle, purely by following gradients downhill.");
    println!();
    println!("🎓 CONGRATULATIONS — you built an autograd engine and trained a neural net");
    println!("   with it, from nothing. Now reopen handcrafted_transformer and read");
    println!("   `out.loss.backward()` again. Same engine. No more magic. 🧠");
}
