use chora::{CpuBackend, Tensor, TensorError};

#[test]
fn zeros_has_correct_shape_and_values() {
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 3]);
    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);
    assert_eq!(t.get(&[1, 2]).unwrap(), 0.0);
}

#[test]
fn ones_has_correct_shape_and_values() {
    let t: Tensor<CpuBackend> = Tensor::ones(&[2, 2]);
    assert_eq!(t.shape(), &[2, 2]);
    assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(t.get(&[1, 1]).unwrap(), 1.0);
}

#[test]
fn from_vec_builds_correct_tensor() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(t.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(t.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(t.get(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn from_vec_rejects_data_shape_mismatch() {
    let result: Result<Tensor<CpuBackend>, TensorError> =
        Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        result,
        Err(TensorError::DataShapeMismatch { data: 3, .. })
    ));
}

#[test]
fn set_updates_value_in_place() {
    let mut t: Tensor<CpuBackend> = Tensor::zeros(&[2, 2]);
    t.set(&[1, 0], 5.0).unwrap();
    assert_eq!(t.get(&[1, 0]).unwrap(), 5.0);
    // other elements should be untouched
    assert_eq!(t.get(&[0, 0]).unwrap(), 0.0);
}

#[test]
fn get_rejects_out_of_bounds_index() {
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 2]);
    let result = t.get(&[2, 0]);
    assert!(matches!(result, Err(TensorError::OutOfBounds { .. })));
}

#[test]
fn get_rejects_wrong_number_of_dims() {
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 2]);
    let result = t.get(&[0, 0, 0]);
    assert!(matches!(result, Err(TensorError::OutOfBounds { .. })));
}

#[test]
fn set_rejects_out_of_bounds_index() {
    let mut t: Tensor<CpuBackend> = Tensor::zeros(&[2, 2]);
    let result = t.set(&[5, 5], 1.0);
    assert!(matches!(result, Err(TensorError::OutOfBounds { .. })));
}

#[test]
fn ravel_matches_row_major_layout() {
    // shape [2, 3] -> strides [3, 1]
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 3]);
    assert_eq!(t.ravel(&[0, 0]), 0);
    assert_eq!(t.ravel(&[0, 2]), 2);
    assert_eq!(t.ravel(&[1, 0]), 3);
    assert_eq!(t.ravel(&[1, 2]), 5);
}

#[test]
fn unravel_is_inverse_of_ravel() {
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 3]);
    for row in 0..2 {
        for col in 0..3 {
            let flat = t.ravel(&[row, col]);
            assert_eq!(t.unravel(flat), vec![row, col]);
        }
    }
}

#[test]
fn negative_value() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let neg_t = chora::neg(&t);
    assert_eq!(neg_t.get(&[0, 0]).unwrap(), -1.0);
    assert_eq!(neg_t.get(&[0, 1]).unwrap(), -2.0);
    assert_eq!(neg_t.get(&[1, 0]).unwrap(), -3.0);
    assert_eq!(neg_t.get(&[1, 1]).unwrap(), -4.0);
}

#[test]
fn absolute_value() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let abs_t = chora::abs(&t);
    assert_eq!(abs_t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(abs_t.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(abs_t.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(abs_t.get(&[1, 1]).unwrap(), 4.0);
}

// write tests for rest of unary ops
