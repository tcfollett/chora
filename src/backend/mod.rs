// Define backend trait
// cpu and gpu should implement backend
// associated storages

pub trait Backend: Clone + Default {
    type Storage: Clone;

    fn zeros(shape: &[usize]) -> Self::Storage;
}
