// Tensor struct

// tensor struct will hold shape, strides, and storage (from backend)
// also have contructors like zeros(), ones(), new()
// trait impls like Clone, Debug, PartialEq, and Display
use crate::backend::Backend;

pub struct Tensor<B: Backend> {
    shape: Vec<usize>,
    strides: Vec<usize>,
    data: B::Storage,
    backend: B,
}

fn strides(shape: &[usize]) -> Vec<usize> {
    let mut stride_list = Vec::with_capacity(shape.len());
    let mut current_stride = 1;
    for i in shape.iter().rev() {
        stride_list.push(current_stride);
        current_stride *= i;
    }
    stride_list.reverse();
    stride_list
}

impl<B: Backend> Tensor<B> {
    // returns shape of tensor
    fn shape(&self) -> &[usize] {
        &self.shape
    }

    // converts a multidim index into a flat index
    fn ravel(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }

    // converts a flat index into multidim index
    fn unravel(&self, flat_index: usize) -> Vec<usize> {
        let mut remaining_value = flat_index;
        let mut index: Vec<usize> = Vec::new();

        for i in self.strides.iter() {
            index.push(remaining_value / i);
            remaining_value %= i;
        }
        index
    }

    // creates a new tensor filled with zeros for the given shape
    fn zeros(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            strides: strides(shape),
            data: B::zeros(shape),
            backend: B::default(),
        }
    }

    // gets a element from a tensor
    fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let out_of_bounds = index.iter().zip(self.shape.iter()).any(|(i, s)| i >= s);
        if index.len() != self.shape.len() {
            return Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        } else if out_of_bounds {
            return Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        } else {
            let flat_index = self.ravel(index);
            Ok(B::read_element(&self.data, flat_index))
        }
    }

    // writes an value to an element of a tensor
    fn set(&mut self, index: &[usize], value: f32) -> Result<(), TensorError> {
        let out_of_bounds = index.iter().zip(self.shape.iter()).any(|(i, s)| i >= s);
        if index.len() != self.shape.len() {
            return Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        } else if out_of_bounds {
            return Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            });
        } else {
            let flat_index = self.ravel(index);
            Ok(B::write_element(&mut self.data, flat_index, value))
        }
    }

    // creates a new tensor filled with ones for the given shape
    fn ones(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            strides: strides(shape),
            data: B::ones(shape),
            backend: B::default(),
        }
    }
}
