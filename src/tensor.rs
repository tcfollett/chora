use crate::shape::TensorShape;
use crate::storage::TensorStorage;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: TensorShape) -> Self {
        let tensor_size = shape.size();
        let tensor_storage = TensorStorage::zeros(tensor_size);
        Tensor {
            shape,
            storage: tensor_storage,
        }
    }
}

