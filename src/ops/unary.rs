use crate::{backend::Backend, tensor::Tensor};

// returns a new tensor with each element negated
pub fn neg<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| -x);
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the absolute value of each element
pub fn abs<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| x.abs());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the sqrt of each element
pub fn sqrt<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| x.sqrt());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the natural log of each element
pub fn ln<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| x.ln());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the exponential of each element
pub fn exp<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| x.exp());
    Tensor::from_storage(tensor.shape(), new_data)
}

// returns a new tensor with the square of each element
pub fn square<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| x * x);
    Tensor::from_storage(tensor.shape(), new_data)
}
