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
}
