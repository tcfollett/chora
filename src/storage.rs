#[derive(Debug, Clone)]
pub struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> TensorStorage<T> {
    pub fn zeros() {}
}
