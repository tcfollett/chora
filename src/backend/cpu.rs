// cpu backend

use crate::backend::Backend;
use crate::tensor::strides;

#[derive(Clone, Default, Debug, PartialEq)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Storage = Vec<f32>;

    fn length(storage: &Self::Storage) -> usize {
        storage.len()
    }

    fn zeros(shape: &[usize]) -> Vec<f32> {
        vec![0.0; shape.iter().product()]
    }

    fn read_element(storage: &Self::Storage, index: usize) -> f32 {
        storage[index]
    }

    fn write_element(storage: &mut Self::Storage, index: usize, value: f32) {
        storage[index] = value;
    }

    fn ones(shape: &[usize]) -> Vec<f32> {
        vec![1.0; shape.iter().product()]
    }

    fn from_vec(data: Vec<f32>) -> Vec<f32> {
        data
    }

    // unary ops

    fn neg(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| -x).collect()
    }

    fn abs(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| x.abs()).collect()
    }

    fn sqrt(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| x.sqrt()).collect()
    }

    fn square(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| x * x).collect()
    }

    fn ln(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| x.ln()).collect()
    }

    fn exp(storage: &Self::Storage) -> Vec<f32> {
        storage.iter().map(|x| x.exp()).collect()
    }

    // elementwise ops

    fn add(a: &Self::Storage, b: &Self::Storage) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(a, b)| a + b).collect()
    }

    fn sub(a: &Self::Storage, b: &Self::Storage) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(a, b)| a - b).collect()
    }

    fn mult(a: &Self::Storage, b: &Self::Storage) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(a, b)| a * b).collect()
    }

    fn div(a: &Self::Storage, b: &Self::Storage) -> Vec<f32> {
        a.iter().zip(b.iter()).map(|(a, b)| a / b).collect()
    }

    // reduction ops

    // this needs to be optimized
    fn sum(storage: &Self::Storage, shape: &[usize], axis: Option<usize>) -> Vec<f32> {
        match axis {
            Some(ax) => {
                let mut output_shape: Vec<usize> = Vec::new();
                for (idx, val) in shape.iter().enumerate() {
                    if idx != ax {
                        output_shape.push(*val);
                    }
                }

                let output_strides = strides(&output_shape);
                let input_strides = strides(shape);
                let output_len = output_shape.iter().product();
                let mut output = vec![0.0; output_len];

                for i in 0..storage.len() {
                    let mut remaining_value = i;
                    let mut index: Vec<usize> = Vec::new();

                    for stride in input_strides.iter() {
                        index.push(remaining_value / stride);
                        remaining_value %= stride;
                    }

                    let mut output_index: Vec<usize> = Vec::new();
                    for (idx, val) in index.iter().enumerate() {
                        if idx != ax {
                            output_index.push(*val);
                        }
                    }

                    let output_flat_index: usize = output_index
                        .iter()
                        .zip(output_strides.iter())
                        .map(|(a, b)| *a * *b)
                        .sum();
                    output[output_flat_index] += storage[i];
                }
                output
            }
            None => {
                vec![storage.iter().sum()]
            }
        }
    }

    fn mean(storage: &Self::Storage, shape: &[usize], axis: Option<usize>) -> Vec<f32> {
        let sum_values = Self::sum(storage, shape, axis);

        match axis {
            Some(ax) => {
                let element_count = shape[ax] as f32;
                sum_values.iter().map(|x| x / element_count).collect()
            }
            None => {
                let element_count = shape.iter().product::<usize>() as f32;
                sum_values.iter().map(|x| x / element_count).collect()
            }
        }
    }

    fn max(storage: &Self::Storage, shape: &[usize], axis: Option<usize>) -> Vec<f32> {
        match axis {
            Some(ax) => {
                let mut output_shape: Vec<usize> = Vec::new();
                for (idx, val) in shape.iter().enumerate() {
                    if idx != ax {
                        output_shape.push(*val);
                    }
                }

                let output_strides = strides(&output_shape);
                let input_strides = strides(shape);
                let output_len = output_shape.iter().product();
                let mut output = vec![f32::MIN; output_len];

                for i in 0..storage.len() {
                    let mut remaining_value = i;
                    let mut index: Vec<usize> = Vec::new();

                    for stride in input_strides.iter() {
                        index.push(remaining_value / stride);
                        remaining_value %= stride;
                    }

                    let mut output_index: Vec<usize> = Vec::new();
                    for (idx, val) in index.iter().enumerate() {
                        if idx != ax {
                            output_index.push(*val);
                        }
                    }

                    let output_flat_index: usize = output_index
                        .iter()
                        .zip(output_strides.iter())
                        .map(|(a, b)| *a * *b)
                        .sum();
                    output[output_flat_index] = output[output_flat_index].max(storage[i]);
                }
                output
            }
            None => {
                vec![storage.iter().fold(f32::MIN, |acc, x| acc.max(*x))]
            }
        }
    }

    fn min(storage: &Self::Storage, shape: &[usize], axis: Option<usize>) -> Vec<f32> {
        match axis {
            Some(ax) => {
                let mut output_shape: Vec<usize> = Vec::new();
                for (idx, val) in shape.iter().enumerate() {
                    if idx != ax {
                        output_shape.push(*val);
                    }
                }

                let output_strides = strides(&output_shape);
                let input_strides = strides(shape);
                let output_len = output_shape.iter().product();
                let mut output = vec![f32::MAX; output_len];

                for i in 0..storage.len() {
                    let mut remaining_value = i;
                    let mut index: Vec<usize> = Vec::new();

                    for stride in input_strides.iter() {
                        index.push(remaining_value / stride);
                        remaining_value %= stride;
                    }

                    let mut output_index: Vec<usize> = Vec::new();
                    for (idx, val) in index.iter().enumerate() {
                        if idx != ax {
                            output_index.push(*val);
                        }
                    }

                    let output_flat_index: usize = output_index
                        .iter()
                        .zip(output_strides.iter())
                        .map(|(a, b)| *a * *b)
                        .sum();
                    output[output_flat_index] = output[output_flat_index].min(storage[i]);
                }
                output
            }
            None => {
                vec![storage.iter().fold(f32::MAX, |acc, x| acc.min(*x))]
            }
        }
    }

    // matmul

    fn matmul(
        a: &Self::Storage,
        b: &Self::Storage,
        a_shape: &[usize],
        b_shape: &[usize],
    ) -> Vec<f32> {
        let m = a_shape[0];
        let n = a_shape[1];
        let p = b_shape[1];
        let mut output = vec![0.0; m * p];

        for i in 0..m {
            for j in 0..p {
                for k in 0..n {
                    output[i * p + j] += a[i * n + k] * b[k * p + j];
                }
            }
        }
        output
    }
}
