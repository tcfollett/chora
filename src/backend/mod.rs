// Define backend trait
// cpu and gpu should implement backend
// associated storages

pub mod cpu;

pub trait Backend: Clone + Default {
    type Storage: Clone;

    fn zeros(shape: &[usize]) -> Self::Storage;

    fn read_element(storage: &Self::Storage, index: usize) -> f32;

    fn write_element(storage: &mut Self::Storage, index: usize, value: f32);

    fn ones(shape: &[usize]) -> Self::Storage;

    fn from_vec(data: Vec<f32>) -> Self::Storage;

    fn map(storage: &Self::Storage, function: impl Fn(f32) -> f32) -> Self::Storage;

    fn binary_map(
        a: &Self::Storage,
        b: &Self::Storage,
        function: impl Fn(f32, f32) -> f32,
    ) -> Self::Storage;
}
