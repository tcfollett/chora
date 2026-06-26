#[derive(Debug, Clone)]
pub struct TensorStorage<T> {
    data: Vec<T>,
}

impl<T> TensorStorage<T> {
    pub fn new(data: Vec<T>) -> Self {
        Self { data }
    }

    pub fn read_data(&self, index: usize) -> &T {
        &self.data[index]
    }
}

impl<T: Default + Clone> TensorStorage<T> {
    pub fn zeros(size: usize) -> Self {
        let zeros = vec![T::default(); size];
        Self { data: zeros }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero() {
        let storage: TensorStorage<i32> = TensorStorage::zeros(3);
        assert_eq!(storage.data.len(), 3);
        assert_eq!(storage.data, vec![0, 0, 0]);
    }
}
