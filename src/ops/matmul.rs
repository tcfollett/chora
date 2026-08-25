use crate::{backend::Backend, error::TensorError, tensor::Tensor};

// performs matrix multiplication between two rank-two tensors
// currently panics if a or b is not 2-dimensional
pub fn matmul<B: Backend>(a: &Tensor<B>, b: &Tensor<B>) -> Result<Tensor<B>, TensorError> {
    if a.shape()[1] != b.shape()[0] {
        Err(TensorError::ShapeMismatch {
            shape1: a.shape().to_vec(),
            shape2: b.shape().to_vec(),
        })
    } else {
        let new_data = B::matmul(a.storage(), b.storage(), a.shape(), b.shape());
        let output_shape = vec![a.shape()[0], b.shape()[1]];
        Tensor::from_storage(&output_shape, new_data)
    }
}
