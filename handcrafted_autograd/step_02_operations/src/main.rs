//! Step 2: Remembering Your Parents (Operations & the Graph)
//!
//! Run from inside this folder with:
//!   cargo run
//!
//! Analogy:
//!   When you compute `e = a * b`, the answer `e` should remember that it was
//!   born from `a` TIMES `b`. We teach `+` and `*` to record two things:
//!     • prev → the parent values that made this one
//!     • op   → which operation made it ("+", "*", ...)
//!   String many of these together and you've drawn a family tree:
//!   the COMPUTATION GRAPH. (In Step 3 we walk this tree backwards to get slopes.)
//!
//! Burn bridge:
//!   Burn builds this same graph for the transformer automatically when you
//!   write `q.matmul(k_t)` etc. It has to, so that `loss.backward()` knows the
//!   path back to every weight. Here we build the graph by hand for `+` and `*`.

#![allow(dead_code)]

use std::cell::RefCell;
use std::fmt;
use std::ops::{Add, Mul};
use std::rc::Rc;

#[derive(Clone)]
pub struct Value(Rc<RefCell<Inner>>);

struct Inner {
    data: f64,
    grad: f64,
    op: &'static str,   // the operation that produced this value
    prev: Vec<Value>,   // the parents (inputs) of this value
    label: String,
}

impl Value {
    pub fn new(data: f64, label: &str) -> Value {
        Value(Rc::new(RefCell::new(Inner {
            data,
            grad: 0.0,
            op: "leaf",
            prev: vec![],
            label: label.to_string(),
        })))
    }

    pub fn data(&self) -> f64 {
        self.0.borrow().data
    }

    /// Pretty-print the whole family tree underneath this value.
    pub fn print_graph(&self) {
        self.print_indented(0);
    }

    fn print_indented(&self, depth: usize) {
        let inner = self.0.borrow();
        let pad = "    ".repeat(depth);
        let name = if inner.label.is_empty() { "·" } else { &inner.label };
        println!(
            "{pad}└─ {name:<6} = {:+.4}   (made by '{}')",
            inner.data, inner.op
        );
        for parent in inner.prev.iter() {
            parent.print_indented(depth + 1);
        }
    }
}

/// `a + b` — and the result remembers its two parents and the "+" op.
impl Add for &Value {
    type Output = Value;
    fn add(self, rhs: &Value) -> Value {
        Value(Rc::new(RefCell::new(Inner {
            data: self.0.borrow().data + rhs.0.borrow().data,
            grad: 0.0,
            op: "+",
            prev: vec![self.clone(), rhs.clone()],
            label: String::new(),
        })))
    }
}

/// `a * b` — same idea, with the "*" op.
impl Mul for &Value {
    type Output = Value;
    fn mul(self, rhs: &Value) -> Value {
        Value(Rc::new(RefCell::new(Inner {
            data: self.0.borrow().data * rhs.0.borrow().data,
            grad: 0.0,
            op: "*",
            prev: vec![self.clone(), rhs.clone()],
            label: String::new(),
        })))
    }
}

impl fmt::Display for Value {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Value(data={:+.4}, op='{}')", self.data(), self.0.borrow().op)
    }
}

fn main() {
    println!("➕ STEP 2: REMEMBERING YOUR PARENTS (THE GRAPH)");
    println!("===============================================");
    println!("Every operation records WHO made it. That family tree is the graph.");
    println!();

    // Build the expression:   d = a * b + c
    let a = Value::new(2.0, "a");
    let b = Value::new(-3.0, "b");
    let c = Value::new(10.0, "c");

    let e = &a * &b; // e = a * b  = -6
    let d = &e + &c; // d = e + c  =  4

    println!("🧮 We built:  d = a * b + c");
    println!("   a = {:+.1},  b = {:+.1},  c = {:+.1}", a.data(), b.data(), c.data());
    println!("   e = a * b = {:+.1}", e.data());
    println!("   d = e + c = {:+.1}", d.data());
    println!();

    println!("🌳 The computation graph underneath `d`:");
    d.print_graph();
    println!();

    println!("💡 What just happened?");
    println!("   `d` knows it came from '+' of (e, c).");
    println!("   `e` knows it came from '*' of (a, b).");
    println!("   Follow the arrows DOWN to compute the answer (the 'forward pass').");
    println!("   In Step 3 we follow them back UP to compute slopes (backprop).");
    println!();
    println!("🎉 Step 2 complete! The graph is built — time to flow gradients through it.");
}
