use crate::shape::TensorShape;
use crate::storage::TensorStorage;
use core::fmt;
use std::{
    fmt::Formatter,
    ops::{Index, IndexMut},
};

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

impl<T: fmt::Display> fmt::Display for Tensor<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        Tensor::print(self.shape.dimensions(), self.storage.slice(), f)
    }
}

impl<T: fmt::Display> Tensor<T> {
    // i need to see if I can simplify this
    fn print(shape: &[usize], data_slice: &[T], f: &mut Formatter<'_>) -> fmt::Result {
        if shape.len() == 1 {
            write!(f, "[")?;
            for (idx, val) in data_slice.iter().enumerate() {
                if idx != 0 {
                    write!(f, ", {}", val)?;
                } else {
                    write!(f, "{}", val)?;
                }
            }
            write!(f, "]")?;
            Ok(())
        } else {
            let chunk_size = shape[1..].iter().product();
            for chunk in data_slice.chunks(chunk_size) {
                Ok(())
            }
        }
    }
}
