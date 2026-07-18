// storage will be a plain vec
// start with loops and look into rayon later

use crate::{TensorError, backend::Backend};

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
}
