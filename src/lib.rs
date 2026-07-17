// public API

mod backend;
mod error;
mod ops;
mod tensor;

pub use backend::cpu::CpuBackend;
pub use error::TensorError;
pub use ops::{abs, exp, ln, neg, sqrt, square};
pub use tensor::Tensor;
