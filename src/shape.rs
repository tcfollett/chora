#[derive(Debug, Clone, PartialEq)]
pub struct TensorShape {
    shape: Vec<usize>,
}

impl TensorShape {
    pub fn new(shape: Vec<usize>) -> Self {
        TensorShape { shape }
    }

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

    pub fn ravel(&self, index: &Vec<usize>) -> usize {
        index
            .iter()
            .zip(self.strides().iter())
            .map(|(a, b)| *a * *b)
            .sum()
    }

    pub fn unravel(&self, flat_index: usize) -> Vec<usize> {
        let mut remaining_value = flat_index;
        let mut index: Vec<usize> = Vec::new();

        for i in self.strides().iter() {
            index.push(remaining_value / i);
            remaining_value %= i;
        }
        index
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

    #[test]
    fn ravel_2x3x4() {
        let shape = TensorShape {
            shape: vec![2, 3, 4],
        };
        assert_eq!(shape.ravel(&vec![1, 2, 3]), 23);
        assert_eq!(shape.ravel(&vec![0, 1, 2]), 6);
        assert_eq!(shape.ravel(&vec![0, 0, 0]), 0);
    }

    #[test]
    fn unravel_2x3x4() {
        let shape = TensorShape {
            shape: vec![2, 3, 4],
        };
        assert_eq!(shape.unravel(23), vec![1, 2, 3]);
        assert_eq!(shape.unravel(6), vec![0, 1, 2]);
        assert_eq!(shape.unravel(0), vec![0, 0, 0]);
    }
}
