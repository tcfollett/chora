// Tensor struct

use crate::{backend::Backend, error::TensorError};
use core::fmt;
use std::fmt::Formatter;

#[derive(Clone, Default, Debug, PartialEq)]
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
    // returns storage of tensor
    pub(crate) fn storage(&self) -> &B::Storage {
        &self.data
    }

    // makes a tensor from existing shape and data
    pub(crate) fn from_storage(shape: &[usize], data: B::Storage) -> Result<Self, TensorError> {
        if B::length(&data) != shape.iter().product() {
            Err(TensorError::DataShapeMismatch {
                data: B::length(&data),
                shape: shape.to_vec(),
            })
        } else {
            Ok(Tensor {
                shape: shape.to_vec(),
                strides: strides(shape),
                data,
                backend: B::default(),
            })
        }
    }

    // returns shape of tensor
    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    // converts a multidim index into a flat index
    pub fn ravel(&self, index: &[usize]) -> usize {
        index
            .iter()
            .zip(self.strides.iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }

    // converts a flat index into multidim index
    pub fn unravel(&self, flat_index: usize) -> Vec<usize> {
        let mut remaining_value = flat_index;
        let mut index: Vec<usize> = Vec::new();

        for i in self.strides.iter() {
            index.push(remaining_value / i);
            remaining_value %= i;
        }
        index
    }

    // creates a new tensor filled with zeros for the given shape
    pub fn zeros(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            strides: strides(shape),
            data: B::zeros(shape),
            backend: B::default(),
        }
    }

    // gets a element from a tensor
    pub fn get(&self, index: &[usize]) -> Result<f32, TensorError> {
        let out_of_bounds = index.iter().zip(self.shape.iter()).any(|(i, s)| i >= s);
        if index.len() != self.shape.len() {
            Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            })
        } else if out_of_bounds {
            Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            })
        } else {
            let flat_index = self.ravel(index);
            Ok(B::read_element(&self.data, flat_index))
        }
    }

    // writes an value to an element of a tensor
    pub fn set(&mut self, index: &[usize], value: f32) -> Result<(), TensorError> {
        let out_of_bounds = index.iter().zip(self.shape.iter()).any(|(i, s)| i >= s);
        if index.len() != self.shape.len() {
            Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            })
        } else if out_of_bounds {
            Err(TensorError::OutOfBounds {
                index: index.to_vec(),
                shape: self.shape.clone(),
            })
        } else {
            let flat_index = self.ravel(index);
            Ok(B::write_element(&mut self.data, flat_index, value))
        }
    }

    // creates a new tensor filled with ones for the given shape
    pub fn ones(shape: &[usize]) -> Self {
        Tensor {
            shape: shape.to_vec(),
            strides: strides(shape),
            data: B::ones(shape),
            backend: B::default(),
        }
    }

    // creates a new tensor filled with data from a vec
    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Result<Self, TensorError> {
        if data.len() != shape.iter().product() {
            Err(TensorError::DataShapeMismatch {
                data: data.len(),
                shape: shape.to_vec(),
            })
        } else {
            Ok(Tensor {
                shape: shape.to_vec(),
                strides: strides(shape),
                data: B::from_vec(data),
                backend: B::default(),
            })
        }
    }

    // changes the shape of an existing tensor
    pub fn reshape(&self, new_shape: &[usize]) -> Result<Self, TensorError> {
        if B::length(&self.data) != new_shape.iter().product() {
            Err(TensorError::DataShapeMismatch {
                data: B::length(&self.data),
                shape: new_shape.to_vec(),
            })
        } else {
            Ok(Tensor {
                shape: new_shape.to_vec(),
                strides: strides(new_shape),
                data: self.data.clone(),
                backend: B::default(),
            })
        }
    }

    // prints a tensor
    fn print(&self, f: &mut Formatter<'_>) -> fmt::Result {
        let total_elements = self.shape().iter().product();
        for j in 0..total_elements {
            let index = self.unravel(j);
            let open_count = index.iter().rev().take_while(|&&x| x == 0).count();
            let close_count = index
                .iter()
                .zip(self.shape().iter())
                .rev()
                .take_while(|&(&i, &s)| i == s - 1)
                .count();
            let value = self.get(&index).unwrap();
            write!(f, "{}", "[".repeat(open_count))?;
            write!(f, "{}", value)?;
            write!(f, "{}", "]".repeat(close_count))?;
            if j != total_elements - 1 {
                write!(f, ", ")?;
            }
        }
        Ok(())
    }
}

impl<B: Backend> fmt::Display for Tensor<B> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        self.print(f)
    }
}
