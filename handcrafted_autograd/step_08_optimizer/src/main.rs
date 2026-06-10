//! Step 8: Building Adam (The Optimizer)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   Plain gradient descent (Step 7) is a hiker taking equal-sized steps.
//!   ADAM is a smarter hiker with two upgrades:
//!     • momentum   → keep some speed from previous steps (roll through bumps).
//!     • adaptive   → big grads get smaller steps, tiny grads get bigger steps,
//!                    so every weight moves at a sensible pace.
//!   This is the exact `AdamConfig` the transformer trained with.
//!
//! An "optimizer" is just an object that owns the update rule. We give it the
//! same shape as Burn's: build it, then call `.step(&params)` each iteration.
//!
//! (ENGINE + NN + LOSS unchanged. The OPTIM block is new.)

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
//  THE LOSS  (MSE — identical to Step 6)
// ============================================================================

pub fn mse(preds: &[Value], targets: &[f64]) -> Value {
    let mut loss = Value::new(0.0);
    for (p, t) in preds.iter().zip(targets.iter()) {
        let diff = p - &Value::new(*t);
        loss = &loss + &diff.powf(2.0);
    }
    &loss * &Value::new(1.0 / preds.len() as f64)
}

// ============================================================================
//  THE OPTIMIZERS  —  the update rules, as objects
// ============================================================================

/// Plain Stochastic Gradient Descent: the Step-7 rule, packaged.
pub struct Sgd {
    lr: f64,
}
impl Sgd {
    pub fn new(lr: f64) -> Self {
        Self { lr }
    }
    pub fn zero_grad(&self, params: &[Value]) {
        for p in params {
            p.zero_grad();
        }
    }
    pub fn step(&self, params: &[Value]) {
        for p in params {
            p.set_data(p.data() - self.lr * p.grad());
        }
    }
}

/// Adam: momentum (`m`) + adaptive per-weight scaling (`v`), with bias
/// correction so the very first steps aren't tiny. Defaults match Burn / PyTorch.
pub struct Adam {
    lr: f64,
    beta1: f64, // momentum decay
    beta2: f64, // variance decay
    eps: f64,   // avoids divide-by-zero
    t: i32,     // step counter (for bias correction)
    m: Vec<f64>, // running mean of grads, per parameter
    v: Vec<f64>, // running mean of grad², per parameter
}
impl Adam {
    pub fn new(lr: f64) -> Self {
        Self { lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, t: 0, m: vec![], v: vec![] }
    }
    pub fn zero_grad(&self, params: &[Value]) {
        for p in params {
            p.zero_grad();
        }
    }
    pub fn step(&mut self, params: &[Value]) {
        // Lazily size the state to match the parameter list on first use.
        if self.m.len() != params.len() {
            self.m = vec![0.0; params.len()];
            self.v = vec![0.0; params.len()];
        }
        self.t += 1;
        for (i, p) in params.iter().enumerate() {
            let g = p.grad();
            // 1. update momentum and variance estimates
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g;
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * g * g;
            // 2. bias-correct (early steps would otherwise be too small)
            let m_hat = self.m[i] / (1.0 - self.beta1.powi(self.t));
            let v_hat = self.v[i] / (1.0 - self.beta2.powi(self.t));
            // 3. step: scale each weight's move by 1/sqrt(its own variance)
            p.set_data(p.data() - self.lr * m_hat / (v_hat.sqrt() + self.eps));
        }
    }
}

// ============================================================================
//  DEMO  —  same dataset, now trained by Adam
// ============================================================================

fn forward_dataset(model: &Mlp, xs: &[[f64; 3]]) -> Vec<Value> {
    xs.iter()
        .map(|row| {
            let input: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
            model.forward(&input)[0].clone()
        })
        .collect()
}

fn main() {
    println!("🏎️  STEP 8: BUILDING ADAM (THE OPTIMIZER)");
    println!("=========================================");
    println!("An optimizer owns the update rule. Adam adds momentum + adaptive steps.");
    println!();

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];
    let epochs = 30;

    // Train two identical models so we can compare optimizers fairly.
    let mut rng = StdRng::seed_from_u64(42);
    let model_sgd = Mlp::new(3, &[4, 4, 1], &mut rng);
    let mut rng = StdRng::seed_from_u64(42);
    let model_adam = Mlp::new(3, &[4, 4, 1], &mut rng);

    let sgd = Sgd::new(0.1);
    let mut adam = Adam::new(0.1);

    println!("🔁 Training {epochs} steps — SGD (lr 0.1) vs Adam (lr 0.1):");
    println!("   step |     SGD loss |    Adam loss");
    println!("   -----+--------------+-------------");
    for step in 0..epochs {
        // --- SGD model ---
        let preds = forward_dataset(&model_sgd, &xs);
        let loss_sgd = mse(&preds, &ys);
        sgd.zero_grad(&model_sgd.parameters());
        loss_sgd.backward();
        sgd.step(&model_sgd.parameters());

        // --- Adam model ---
        let preds = forward_dataset(&model_adam, &xs);
        let loss_adam = mse(&preds, &ys);
        adam.zero_grad(&model_adam.parameters());
        loss_adam.backward();
        adam.step(&model_adam.parameters());

        if step % 3 == 0 || step == epochs - 1 {
            println!("   {:>4} | {:>12.5} | {:>12.5}", step, loss_sgd.data(), loss_adam.data());
        }
    }
    println!();

    println!("💡 Adam usually dives faster early on, because each weight gets a");
    println!("   step size tuned to its own gradient history — no hand-tuned schedule.");
    println!("   This is the very optimizer the transformer used: AdamConfig::new().");
    println!();
    println!("🎉 Step 8 complete! We have a real optimizer. Next: a clean training loop.");
}
