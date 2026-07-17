// storage will be a plain vec
// start with loops and look into rayon later

use crate::backend::Backend;

#[derive(Clone, Default)]
pub struct CpuBackend;

impl Backend for CpuBackend {
    type Storage = Vec<f32>;

    fn zeros(shape: &[usize]) -> Vec<f32> {
        vec![0.0; shape.iter().product()]
    }

    fn read_element(storage: &Self::Storage, index: usize) -> f32 {
        storage[index]
    }

    fn write_element(storage: &mut Self::Storage, index: usize, value: f32) {
        storage[index] = value;
    }
}
