#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    pub fn size(&self) -> usize {
        self.shape.iter().product()
    }

    fn strides(&self) -> Vec<usize> {
        let mut stride_list: Vec<usize> = Vec::new();
        let mut current_stride = 1;
        for i in self.shape.iter().rev() {
            stride_list.push(current_stride);
            current_stride *= i;
        }
        stride_list.reverse();
        stride_list
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

    #[test]
    fn strides_of_2x3x4() {
        let shape = TensorShape {
            shape: vec![2, 3, 4],
        };
        assert_eq!(shape.strides(), vec![12, 4, 1]);
    }
}
