// public API

mod backend;
mod error;
mod ops;
mod tensor;

pub use backend::cpu::Cpu;
pub use error::Error;
pub use tensor::Tensor;
