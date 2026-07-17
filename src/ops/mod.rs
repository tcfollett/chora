//havent decided on separate files for sets of operations or each operation having a dedicated file
// ops like add, sub, mult, div, matrix mult, sum, mean, max, min

//each op needs to validate input shape, call backend, wrap result in new tensor

mod elementwise;
mod matmul;
mod reduce;
mod unary;

pub use elementwise::{add, div, mult, sub};
pub use unary::{abs, exp, ln, neg, sqrt, square};
