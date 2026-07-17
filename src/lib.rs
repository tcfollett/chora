// public API

mod backend;
mod error;
mod ops;
mod tensor;

pub use backend::cpu::CpuBackend;
pub use error::TensorError;
pub use ops::{abs, add, div, exp, ln, mult, neg, sqrt, square, sub};
pub use tensor::Tensor;
