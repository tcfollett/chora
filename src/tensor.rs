use crate::shape::TensorShape;
use crate::storage::TensorStorage;

#[derive(Debug, Clone)]
pub struct Tensor<T> {
    shape: TensorShape,
    storage: TensorStorage<T>,
}

impl<T> Tensor<T> {
    pub fn new(shape: TensorShape, data: Vec<T>) -> Self {
        Self {
            shape,
            storage: TensorStorage::new(data),
        }
    }
}

impl<T: Default + Clone> Tensor<T> {
    pub fn zeros(shape: TensorShape) -> Self {
        let tensor_size = shape.size();
        let tensor_storage = TensorStorage::zeros(tensor_size);
        Self {
            shape,
            storage: tensor_storage,
        }
    }
}

