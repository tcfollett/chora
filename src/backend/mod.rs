// Define backend trait
// cpu and gpu should implement backend
// associated storages
pub mod cpu;

pub trait Backend: Clone + Default {
    type Storage: Clone;

    fn length(storage: &Self::Storage) -> usize;

    fn zeros(shape: &[usize]) -> Self::Storage;

    fn read_element(storage: &Self::Storage, index: usize) -> f32;

    fn write_element(storage: &mut Self::Storage, index: usize, value: f32);

    fn ones(shape: &[usize]) -> Self::Storage;

    fn from_vec(data: Vec<f32>) -> Self::Storage;

    // unary ops

    fn neg(storage: &Self::Storage) -> Self::Storage;
    fn abs(storage: &Self::Storage) -> Self::Storage;
    fn sqrt(storage: &Self::Storage) -> Self::Storage;
    fn square(storage: &Self::Storage) -> Self::Storage;
    fn ln(storage: &Self::Storage) -> Self::Storage;
    fn exp(storage: &Self::Storage) -> Self::Storage;

    // elementwise ops

    fn add(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn sub(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn mult(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
    fn div(a: &Self::Storage, b: &Self::Storage) -> Self::Storage;
}
