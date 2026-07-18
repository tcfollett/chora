use crate::{backend::Backend, error::TensorError, tensor::Tensor};

// returns a new tensor with each element negated
pub fn neg<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::neg(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the absolute value of each element
pub fn abs<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::abs(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the sqrt of each element
pub fn sqrt<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::sqrt(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the square of each element
pub fn square<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::square(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the natural log of each element
pub fn ln<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::ln(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the exponential of each element
pub fn exp<B: Backend>(tensor: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    let new_data = B::exp(tensor.storage());
    Tensor::from_storage(tensor.shape(), new_data)
}
