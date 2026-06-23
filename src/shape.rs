#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    pub fn size(&self) -> usize {
        let mut size = 1;
        for i in &self.shape {
            size *= i;
        }
        return size;
    }
}
