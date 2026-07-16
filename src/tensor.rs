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
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}
