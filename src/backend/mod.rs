// Define backend trait
// cpu and gpu should implement backend
// associated storages

pub trait Backend: Clone + Default {
    type Storage: Clone;

    fn zeros(shape: &[usize]) -> Self::Storage;
    fn read_element(storage: &Self::Storage, index: usize) -> f32;
    fn write_element(storage: &mut Self::Storage, index: usize, value: f32);
    fn ones(shape: &[usize]) -> Self::Storage;
}
