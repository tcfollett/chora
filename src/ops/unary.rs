// neg, abs, sqrt, exp, ln
use crate::{backend::Backend, tensor::Tensor};

pub fn neg<B: Backend>(tensor: &Tensor<B>) -> Tensor<B> {
    let new_data = B::map(tensor.storage(), |x| -x);
    Tensor::from_storage(tensor.shape(), new_data)
}
