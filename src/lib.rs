mod backend;
mod error;
mod ops;
mod tensor;

pub use backend::cpu::CpuBackend;
pub use error::TensorError;
pub use ops::{abs, add, div, exp, ln, max, mean, min, mult, neg, sqrt, square, sub, sum};
pub use tensor::Tensor;
