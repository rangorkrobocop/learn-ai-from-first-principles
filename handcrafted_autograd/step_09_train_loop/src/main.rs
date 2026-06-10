//! Step 9: The Tutor (The Training Loop)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   Everything we've built has one natural rhythm:
//!       forward → loss → zero grads → backward → optimizer step
//!   repeated every epoch. We wrap that rhythm in a `Learner` — the same role
//!   Burn's `Learner` plays for the transformer. The tutor drills the student.
//!
//! After this step there is no mystery left in `out.loss.backward()`.
//!
//! (ENGINE + NN + LOSS + OPTIM unchanged. The `Learner` ties them together.)

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
//  THE LEARNER  —  the tutor that owns the training rhythm
// ============================================================================

/// Bundles a model + an optimizer and drills the model on data.
/// This is the role Burn's `Learner` plays for the transformer.
pub struct Learner {
    model: Mlp,
    optim: Adam,
}
impl Learner {
    pub fn new(model: Mlp, optim: Adam) -> Self {
        Self { model, optim }
    }

    /// Run one full forward → loss → zero → backward → step cycle. Returns the loss.
    fn train_step(&mut self, xs: &[[f64; 3]], ys: &[f64]) -> f64 {
        let preds: Vec<Value> = xs
            .iter()
            .map(|row| {
                let input: Vec<Value> = row.iter().map(|&v| Value::new(v)).collect();
                self.model.forward(&input)[0].clone()
            })
            .collect();
        let loss = mse(&preds, ys);

        let params = self.model.parameters();
        self.optim.zero_grad(&params); // 1. clear old slopes
        loss.backward(); //                2. compute new slopes
        self.optim.step(&params); //       3. nudge the weights
        loss.data()
    }

    /// The whole training run: repeat `train_step` for `epochs`, logging progress.
    pub fn fit(&mut self, xs: &[[f64; 3]], ys: &[f64], epochs: usize) {
        for epoch in 0..epochs {
            let loss = self.train_step(xs, ys);
            if epoch % 5 == 0 || epoch == epochs - 1 {
                println!("   epoch {:>3}  loss = {:.6}", epoch, loss);
            }
        }
    }

    pub fn predict(&self, x: &[f64]) -> f64 {
        let input: Vec<Value> = x.iter().map(|&v| Value::new(v)).collect();
        self.model.forward(&input)[0].data()
    }
}

// ============================================================================
//  DEMO
// ============================================================================

fn main() {
    println!("🔁 STEP 9: THE TUTOR (THE TRAINING LOOP)");
    println!("========================================");
    println!("forward → loss → zero_grad → backward → step, wrapped in a `Learner`.");
    println!();

    let xs = [
        [2.0, 3.0, -1.0],
        [3.0, -1.0, 0.5],
        [0.5, 1.0, 1.0],
        [1.0, 1.0, -1.0],
    ];
    let ys = [1.0, -1.0, -1.0, 1.0];

    let mut rng = StdRng::seed_from_u64(42);
    let model = Mlp::new(3, &[4, 4, 1], &mut rng);
    let optim = Adam::new(0.05);
    let mut learner = Learner::new(model, optim);

    println!("🎓 learner.fit(xs, ys, epochs=40):");
    learner.fit(&xs, &ys, 40);
    println!();

    println!("🔮 Final predictions (target in parentheses):");
    for (row, y) in xs.iter().zip(ys.iter()) {
        println!("   {:+.4}  ({:+.1})", learner.predict(row), y);
    }
    println!();

    println!("💡 Compare this to handcrafted_transformer/.../training.rs:");
    println!("   `SupervisedTraining...launch(Learner::new(model, optim, lr))`");
    println!("   is doing EXACTLY this loop — just with tensors and millions of weights.");
    println!();
    println!("🎉 Step 9 complete! The tutor works. Final step: train on real 2D data.");
}
