//! Step 4: One Brain Cell (The Neuron)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   A neuron takes several inputs, multiplies each by a WEIGHT, adds a BIAS,
//!   then squashes the total with tanh:
//!       output = tanh( w1*x1 + w2*x2 + ... + b )
//!   It's a weighted vote followed by a "how excited am I?" knob.
//!
//! We package the loose values from Step 3 into a reusable `Neuron` with a
//! `forward()` method — the same shape Burn gives every layer in the transformer.
//!
//! (The ENGINE block below is identical to Step 3. Scroll past it to the DEMO.)

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

// ============================================================================
//  THE ENGINE  (identical to Step 3 — our scalar autograd)
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
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += (1.0 - t * t) * g;
        }));
        out
    }
    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let y = if x > 0.0 { x } else { 0.0 };
        let out = Value::wrap(Inner { data: y, grad: 0.0, op: "relu", prev: vec![self.clone()], backward: None });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += if x > 0.0 { g } else { 0.0 };
        }));
        out
    }
    pub fn powf(&self, n: f64) -> Value {
        let base = self.0.borrow().data;
        let out = Value::wrap(Inner { data: base.powf(n), grad: 0.0, op: "pow", prev: vec![self.clone()], backward: None });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += n * base.powf(n - 1.0) * g;
        }));
        out
    }
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<Inner>> = HashSet::new();
        fn build(v: &Value, topo: &mut Vec<Value>, visited: &mut HashSet<*const RefCell<Inner>>) {
            if visited.insert(v.id()) {
                for parent in v.0.borrow().prev.iter() {
                    build(parent, topo, visited);
                }
                topo.push(v.clone());
            }
        }
        build(self, &mut topo, &mut visited);
        self.0.borrow_mut().grad = 1.0;
        for node in topo.iter().rev() {
            let g = node.0.borrow().grad;
            let inner = node.0.borrow();
            if let Some(bw) = inner.backward.as_ref() {
                bw(g);
            }
        }
    }
}

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::wrap(Inner { data: self.data() + rhs.data(), grad: 0.0, op: "+", prev: vec![self.clone(), rhs.clone()], backward: None });
        let (a, b) = (self.clone(), rhs.clone());
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += g;
            b.0.borrow_mut().grad += g;
        }));
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
    fn neg(self) -> Value {
        self * &Value::new(-1.0)
    }
}
impl Sub for &Value {
    type Output = Value;
    fn sub(self, rhs: &Value) -> Value {
        self + &(-rhs)
    }
}
impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value(data={:+.4}, grad={:+.4})", self.data(), self.grad())
    }
}

// ============================================================================
//  THE NEURON  —  one brain cell
// ============================================================================

/// `output = tanh( w·x + b )`.
pub struct Neuron {
    w: Vec<Value>, // one weight per input
    b: Value,      // one bias
}

impl Neuron {
    /// Build a neuron from explicit weights + bias (random init arrives in Step 5).
    pub fn new(weights: &[f64], bias: f64) -> Neuron {
        Neuron {
            w: weights.iter().map(|&x| Value::new(x)).collect(),
            b: Value::new(bias),
        }
    }

    /// Forward pass — the weighted vote, then the tanh excitement knob.
    pub fn forward(&self, x: &[Value]) -> Value {
        let mut act = self.b.clone(); // start the running total at the bias
        for (wi, xi) in self.w.iter().zip(x.iter()) {
            act = &act + &(wi * xi); // act += w_i * x_i
        }
        act.tanh()
    }

    /// All the tunable knobs (weights + bias) — exactly what an optimizer nudges.
    pub fn parameters(&self) -> Vec<Value> {
        let mut p = self.w.clone();
        p.push(self.b.clone());
        p
    }
}

// ============================================================================
//  DEMO
// ============================================================================

fn main() {
    println!("🔘 STEP 4: ONE BRAIN CELL (THE NEURON)");
    println!("======================================");
    println!("output = tanh( w1*x1 + w2*x2 + b )  — a weighted vote, then a squash.");
    println!();

    // Same numbers as Step 3, but now tidily packaged inside a Neuron.
    let neuron = Neuron::new(&[-3.0, 1.0], 6.881_373_587_019_543);
    let x = [Value::new(2.0), Value::new(0.0)];

    let out = neuron.forward(&x);
    println!("🧮 Forward pass with inputs x = [2.0, 0.0]:");
    println!("   output = {:+.4}", out.data());
    println!();

    // Learn which knobs to turn: backprop fills every parameter's grad.
    out.backward();

    println!("🌊 After out.backward() — gradient on each knob:");
    for (i, p) in neuron.parameters().iter().enumerate() {
        let name = if i < 2 { format!("w{}", i + 1) } else { "b ".to_string() };
        println!("   {name}.grad = {:+.4}", p.grad());
    }
    println!();

    println!("💡 Each grad says how that knob should move to change the output.");
    println!("   A neuron alone is weak. Next we stack many into a Layer, then an MLP.");
    println!();
    println!("🎉 Step 4 complete! One cell down — let's build a brain.");
}
