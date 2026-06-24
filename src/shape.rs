#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn size_of_2x3() {
        let shape = TensorShape { shape: vec![2, 3] };
        assert_eq!(shape.size(), 6);
    }

    #[test]
    fn size_of_1() {
        let shape = TensorShape { shape: vec![4] };
        assert_eq!(shape.size(), 4);
    }

    #[test]
    fn size_of_0() {
        let shape = TensorShape { shape: vec![] };
        assert_eq!(shape.size(), 1)
    }
}
