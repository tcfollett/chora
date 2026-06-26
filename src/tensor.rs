use crate::shape::TensorShape;
use crate::storage::TensorStorage;
use std::ops::{Index, IndexMut};

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

impl<T> Index<&[usize]> for Tensor<T> {
    type Output = T;
    fn index(&self, index: &[usize]) -> &Self::Output {
        self.storage.read_data(self.shape.ravel(index))
    }
}

impl<T> IndexMut<&[usize]> for Tensor<T> {
    fn index_mut(&mut self, index: &[usize]) -> &mut Self::Output {
        self.storage.write_data(self.shape.ravel(index))
    }
}
