use core::fmt;

#[derive(Debug)]
pub enum TensorError {
    ShapeMismatch {
        shape1: Vec<usize>,
        shape2: Vec<usize>,
    }, // two tensors with incompatible shape
    OutOfBounds {
        index: Vec<usize>,
        shape: Vec<usize>,
    }, // index does not match dimensions of the tensor
    DataShapeMismatch {
        data: usize,
        shape: Vec<usize>,
    }, // data does not fit the shape of the tensor
    InvalidAxis {
        axis: usize,
        shape: Vec<usize>,
    }, // axis numnber does not fit the number of dimensions
}

impl fmt::Display for TensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorError::ShapeMismatch { shape1, shape2 } => {
                write!(
                    f,
                    "Tensor shapes do not match: {:?} vs {:?}",
                    shape1, shape2
                )
            }
            TensorError::OutOfBounds { index, shape } => {
                write!(
                    f,
                    "Index {:?} is out of bounds for shape {:?}",
                    index, shape
                )
            }
            TensorError::DataShapeMismatch { data, shape } => {
                write!(
                    f,
                    "Data length {:?} does not fit tensor shape {:?}",
                    data, shape
                )
            }
            TensorError::InvalidAxis { axis, shape } => {
                write!(f, "Invalid axis {:?} for shape {:?}", axis, shape)
            }
        }
    }
}

impl std::error::Error for TensorError {}
