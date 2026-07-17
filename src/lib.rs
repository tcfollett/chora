// public API

mod backend;
mod error;
mod ops;
mod tensor;

pub use backend::cpu::CpuBackend;
pub use error::TensorError;
pub use tensor::Tensor;
