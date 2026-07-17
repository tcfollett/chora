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
}
