//! Step 3: Pouring Gradients Backwards (Backpropagation)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   Water flows downhill; gradients flow BACKWARDS. We seed the final answer
//!   with a slope of 1.0 ("how does the answer affect itself? exactly 1"),
//!   then walk the graph in reverse. At every node the CHAIN RULE multiplies
//!   the slopes together to pass the gradient on to its parents.
//!
//! Basic mathematics — the chain rule:
//!   If c depends on b, and b depends on a, then
//!       dc/da = dc/db * db/da
//!   "How a affects c" = "how b affects c" times "how a affects b."
//!
//! This file contains the COMPLETE little engine. Steps 4–10 reuse it as-is
//! and build neurons, layers, loss, and Adam on top.
//!
//! Burn bridge:
//!   This `backward()` is the scalar twin of `out.loss.backward()` in
//!   handcrafted_transformer/.../training.rs. Same idea, one number at a time.

#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashSet;
use std::fmt;
use std::ops::{Add, Mul, Neg, Sub};
use std::rc::Rc;

// ============================================================================
//  THE ENGINE  —  a scalar reverse-mode autograd ("micrograd", in Rust)
// ============================================================================

#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);

struct Inner {
    data: f64,
    grad: f64,
    op: &'static str,
    prev: Vec<Value>,
    /// How to send THIS node's gradient back to its parents.
    /// The argument is this node's own grad (`out.grad`); the closure adds the
    /// correct share onto each parent's `grad`. `None` for leaves (no parents).
    backward: Option<Box<dyn Fn(f64)>>,
}

impl Value {
    fn wrap(inner: Inner) -> Value {
        Value(Rc::new(RefCell::new(inner)))
    }

    /// A leaf value (an input or a weight).
    pub fn new(data: f64) -> Value {
        Value::wrap(Inner {
            data,
            grad: 0.0,
            op: "leaf",
            prev: vec![],
            backward: None,
        })
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

    // ---- non-linear squashing functions -----------------------------------

    /// tanh — the classic neuron "excitement" knob, squashes into (-1, 1).
    /// Derivative: d/dx tanh(x) = 1 - tanh(x)^2.
    pub fn tanh(&self) -> Value {
        let t = self.0.borrow().data.tanh();
        let out = Value::wrap(Inner {
            data: t,
            grad: 0.0,
            op: "tanh",
            prev: vec![self.clone()],
            backward: None,
        });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += (1.0 - t * t) * g;
        }));
        out
    }

    /// ReLU — "let positives through, clamp negatives to 0."
    /// Derivative: 1 where x > 0, else 0.
    pub fn relu(&self) -> Value {
        let x = self.0.borrow().data;
        let y = if x > 0.0 { x } else { 0.0 };
        let out = Value::wrap(Inner {
            data: y,
            grad: 0.0,
            op: "relu",
            prev: vec![self.clone()],
            backward: None,
        });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += if x > 0.0 { g } else { 0.0 };
        }));
        out
    }

    /// Raise to a constant power. Derivative: n * x^(n-1).
    pub fn powf(&self, n: f64) -> Value {
        let base = self.0.borrow().data;
        let out = Value::wrap(Inner {
            data: base.powf(n),
            grad: 0.0,
            op: "pow",
            prev: vec![self.clone()],
            backward: None,
        });
        let a = self.clone();
        out.0.borrow_mut().backward = Some(Box::new(move |g| {
            a.0.borrow_mut().grad += n * base.powf(n - 1.0) * g;
        }));
        out
    }

    // ---- the star of the show: backpropagation ----------------------------

    /// Fill in `grad` for this node and every ancestor.
    ///
    /// 1. Sort the graph so parents always come before children (topo-sort).
    /// 2. Seed this node's grad with 1.0 (it affects itself one-for-one).
    /// 3. Walk the order in REVERSE, letting each node hand its gradient
    ///    back to its parents via the chain rule (its `backward` closure).
    pub fn backward(&self) {
        let mut topo: Vec<Value> = vec![];
        let mut visited: HashSet<*const RefCell<Inner>> = HashSet::new();

        fn build(
            v: &Value,
            topo: &mut Vec<Value>,
            visited: &mut HashSet<*const RefCell<Inner>>,
        ) {
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

// ---- operator overloads: build the graph as you do ordinary math ----------

impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        let out = Value::wrap(Inner {
            data: self.data() + rhs.data(),
            grad: 0.0,
            op: "+",
            prev: vec![self.clone(), rhs.clone()],
            backward: None,
        });
        // d(a+b)/da = 1 and d(a+b)/db = 1 → gradient passes straight through.
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
        let out = Value::wrap(Inner {
            data: self.data() * rhs.data(),
            grad: 0.0,
            op: "*",
            prev: vec![self.clone(), rhs.clone()],
            backward: None,
        });
        // d(a*b)/da = b and d(a*b)/db = a → each parent scales by the OTHER's value.
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
//  DEMO  —  Karpathy's famous tanh neuron, gradients checked by hand
// ============================================================================

fn main() {
    println!("🌊 STEP 3: POURING GRADIENTS BACKWARDS (BACKPROP)");
    println!("=================================================");
    println!("We rebuild Andrej Karpathy's classic micrograd neuron and let");
    println!("`backward()` fill in every slope automatically.");
    println!();

    // Inputs and weights (the exact numbers from the micrograd lecture).
    let x1 = Value::new(2.0);
    let x2 = Value::new(0.0);
    let w1 = Value::new(-3.0);
    let w2 = Value::new(1.0);
    let b = Value::new(6.881_373_587_019_543); // a tidy bias that makes outputs round

    // Forward pass:  n = x1*w1 + x2*w2 + b   then   o = tanh(n)
    let x1w1 = &x1 * &w1;
    let x2w2 = &x2 * &w2;
    let n = &(&x1w1 + &x2w2) + &b;
    let o = n.tanh();

    println!("🧮 Forward pass:  o = tanh(x1*w1 + x2*w2 + b)");
    println!("   x1={:+.1} w1={:+.1}   x2={:+.1} w2={:+.1}   b={:+.4}",
        x1.data(), w1.data(), x2.data(), w2.data(), b.data());
    println!("   n (pre-activation) = {:+.4}", n.data());
    println!("   o (output)         = {:+.4}", o.data());
    println!();

    // Backward pass: one call fills in every gradient in the graph.
    o.backward();

    println!("🌊 After o.backward() — gradients computed automatically:");
    println!("   x1.grad = {:+.4}   (expected -1.5000)", x1.grad());
    println!("   w1.grad = {:+.4}   (expected +1.0000)", w1.grad());
    println!("   x2.grad = {:+.4}   (expected +0.5000)", x2.grad());
    println!("   w2.grad = {:+.4}   (expected +0.0000)", w2.grad());
    println!();

    // Verify against the known-correct values from the lecture.
    let ok = (x1.grad() - -1.5).abs() < 1e-4
        && (w1.grad() - 1.0).abs() < 1e-4
        && (x2.grad() - 0.5).abs() < 1e-4
        && (w2.grad() - 0.0).abs() < 1e-4;
    println!("✅ Gradient check: {}", if ok { "PASS — matches the math exactly!" } else { "FAIL" });
    assert!(ok, "gradients did not match the expected micrograd values");
    println!();

    println!("💡 Read x1.grad = -1.5 as:");
    println!("   'nudging x1 up by 1 would LOWER the output by about 1.5.'");
    println!("   Training just flips that around: nudge each weight the way that");
    println!("   improves the score. That's all backprop is for.");
    println!();
    println!("🎉 Step 3 complete! The engine is finished. Now we build a brain on it.");
}
