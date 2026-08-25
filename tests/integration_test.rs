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
    let neg_t = neg_t.unwrap();
    assert_eq!(neg_t.get(&[0, 0]).unwrap(), -1.0);
    assert_eq!(neg_t.get(&[0, 1]).unwrap(), -2.0);
    assert_eq!(neg_t.get(&[1, 0]).unwrap(), -3.0);
    assert_eq!(neg_t.get(&[1, 1]).unwrap(), -4.0);
}

#[test]
fn absolute_value() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![-1.0, 2.0, -3.0, 4.0]).unwrap();
    let abs_t = chora::abs(&t);
    let abs_t = abs_t.unwrap();
    assert_eq!(abs_t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(abs_t.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(abs_t.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(abs_t.get(&[1, 1]).unwrap(), 4.0);
}

// write tests for rest of unary ops

#[test]
fn reshape_preserves_data_and_original_tensor() {
    let t: Tensor<CpuBackend> =
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let reshaped = t.reshape(&[3, 2]).unwrap();

    assert_eq!(reshaped.shape(), &[3, 2]);

    assert_eq!(reshaped.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(reshaped.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(reshaped.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(reshaped.get(&[1, 1]).unwrap(), 4.0);
    assert_eq!(reshaped.get(&[2, 0]).unwrap(), 5.0);
    assert_eq!(reshaped.get(&[2, 1]).unwrap(), 6.0);

    assert_eq!(t.shape(), &[2, 3]);
    assert_eq!(t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(t.get(&[1, 2]).unwrap(), 6.0);
}

#[test]
fn reshape_rejects_element_count_mismatch() {
    let t: Tensor<CpuBackend> =
        Tensor::from_vec(&[2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]).unwrap();

    let result = t.reshape(&[2, 2]);

    assert!(matches!(
        result,
        Err(TensorError::DataShapeMismatch { data: 6, .. })
    ));
}

#[test]
fn square_root() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 4.0, 9.0, 16.0]).unwrap();
    let sqrt_t = chora::sqrt(&t).unwrap();
    assert_eq!(sqrt_t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(sqrt_t.get(&[0, 1]).unwrap(), 2.0);
    assert_eq!(sqrt_t.get(&[1, 0]).unwrap(), 3.0);
    assert_eq!(sqrt_t.get(&[1, 1]).unwrap(), 4.0);
}

#[test]
fn square_root_of_negative_is_nan() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[1], vec![-4.0]).unwrap();
    let sqrt_t = chora::sqrt(&t).unwrap();
    assert!(sqrt_t.get(&[0]).unwrap().is_nan());
}

#[test]
fn square_value() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, -2.0, 3.0, -4.0]).unwrap();
    let square_t = chora::square(&t).unwrap();
    assert_eq!(square_t.get(&[0, 0]).unwrap(), 1.0);
    assert_eq!(square_t.get(&[0, 1]).unwrap(), 4.0);
    assert_eq!(square_t.get(&[1, 0]).unwrap(), 9.0);
    assert_eq!(square_t.get(&[1, 1]).unwrap(), 16.0);
}

#[test]
fn natural_log() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2], vec![1.0, std::f32::consts::E]).unwrap();
    let ln_t = chora::ln(&t).unwrap();
    assert_eq!(ln_t.get(&[0]).unwrap(), 0.0);
    assert!((ln_t.get(&[1]).unwrap() - 1.0).abs() < 1e-6);
}

#[test]
fn natural_log_of_zero_is_negative_infinity() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[1], vec![0.0]).unwrap();
    let ln_t = chora::ln(&t).unwrap();
    assert_eq!(ln_t.get(&[0]).unwrap(), f32::NEG_INFINITY);
}

#[test]
fn exponential() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2], vec![0.0, 1.0]).unwrap();
    let exp_t = chora::exp(&t).unwrap();
    assert_eq!(exp_t.get(&[0]).unwrap(), 1.0);
    assert!((exp_t.get(&[1]).unwrap() - std::f32::consts::E).abs() < 1e-6);
}

#[test]
fn matmul_multiplies_two_matrices() {
    let a: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let b: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![5.0, 6.0, 7.0, 8.0]).unwrap();
    let c = chora::matmul(&a, &b).unwrap();
    assert_eq!(c.shape(), &[2, 2]);
    assert_eq!(c.get(&[0, 0]).unwrap(), 19.0);
    assert_eq!(c.get(&[0, 1]).unwrap(), 22.0);
    assert_eq!(c.get(&[1, 0]).unwrap(), 43.0);
    assert_eq!(c.get(&[1, 1]).unwrap(), 50.0);
}

#[test]
fn get_rejects_out_of_bounds_index2() {
    let t: Tensor<CpuBackend> = Tensor::zeros(&[2, 2]);
    let result = t.get(&[2, 0]);
    assert!(matches!(result, Err(TensorError::OutOfBounds { .. })));
}

#[test]
fn sum_sums_all_elements() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::sum(&t, None).unwrap();
    assert_eq!(result.get(&[0]).unwrap(), 10.0);
}

#[test]
fn mean_returns_average_of_all_elements() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::mean(&t, None).unwrap();
    assert_eq!(result.get(&[0]).unwrap(), 2.5);
}

#[test]
fn max_returns_largest_element() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 4.0, 3.0, 2.0]).unwrap();
    let result = chora::max(&t, None).unwrap();
    assert_eq!(result.get(&[0]).unwrap(), 4.0);
}

#[test]
fn min_returns_smallest_element() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![4.0, 1.0, 3.0, 2.0]).unwrap();
    let result = chora::min(&t, None).unwrap();
    assert_eq!(result.get(&[0]).unwrap(), 1.0);
}

#[test]
fn sum_rejects_invalid_axis() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::sum(&t, Some(5));
    assert!(matches!(result, Err(TensorError::InvalidAxis { .. })));
}

#[test]
fn mean_rejects_invalid_axis() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::mean(&t, Some(5));
    assert!(matches!(result, Err(TensorError::InvalidAxis { .. })));
}

#[test]
fn max_rejects_invalid_axis() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::max(&t, Some(5));
    assert!(matches!(result, Err(TensorError::InvalidAxis { .. })));
}

#[test]
fn min_rejects_invalid_axis() {
    let t: Tensor<CpuBackend> = Tensor::from_vec(&[2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap();
    let result = chora::min(&t, Some(5));
    assert!(matches!(result, Err(TensorError::InvalidAxis { .. })));
}
