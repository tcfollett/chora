use crate::{backend::Backend, error::TensorError, tensor::Tensor};

// adds two tensors
pub fn add<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    if a.shape() != b.shape() {
        Err(TensorError::ShapeMismatch {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        })
    } else {
        let new_data = B::add(a.storage(), b.storage());
        Tensor::from_storage(a.shape(), new_data)
    }
}

// subtracts two tensors
pub fn sub<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    if a.shape() != b.shape() {
        Err(TensorError::ShapeMismatch {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        })
    } else {
        let new_data = B::sub(a.storage(), b.storage());
        Tensor::from_storage(a.shape(), new_data)
    }
}

// multiplies two tensors
pub fn mult<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    if a.shape() != b.shape() {
        Err(TensorError::ShapeMismatch {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        })
    } else {
        let new_data = B::mult(a.storage(), b.storage());
        Tensor::from_storage(a.shape(), new_data)
    }
}

// divides two tensors
pub fn div<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    if a.shape() != b.shape() {
        Err(TensorError::ShapeMismatch {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        })
    } else {
        let new_data = B::div(a.storage(), b.storage());
        Tensor::from_storage(a.shape(), new_data)
    }
}
