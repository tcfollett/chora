use crate::{backend::Backend, error::TensorError, tensor::Tensor};

// reduction ops helper function for reducing tensor shape
fn reduced_shape<B: Backend>(
    tensor: &Tensor<B>,
    axis: Option<usize>,
) -> Result<Vec<usize>, TensorError> {
    if let Some(ax) = axis {
        if ax >= tensor.shape().len() {
            return Err(TensorError::InvalidAxis {
                axis: ax,
                shape: tensor.shape().to_vec(),
            });
        }
    }
    match axis {
        Some(ax) => {
            let mut output_shape: Vec<usize> = Vec::new();
            for (idx, val) in tensor.shape().iter().enumerate() {
                if idx != ax {
                    output_shape.push(*val);
                }
            }
            Ok(output_shape)
        }
        None => Ok(vec![1]),
    }
}

// returns the sum the entire tensor or an axis of the tensor
pub fn sum<B: Backend>(tensor: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, TensorError> {
    let output_shape = reduced_shape(tensor, axis)?;
    let new_data = B::sum(tensor.storage(), tensor.shape(), axis);
    Tensor::from_storage(&output_shape, new_data)
}

// returns the mean of the entire tensor or an axis of the tensor
pub fn mean<B: Backend>(tensor: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, TensorError> {
    let output_shape = reduced_shape(tensor, axis)?;
    let new_data = B::mean(tensor.storage(), tensor.shape(), axis);
    Tensor::from_storage(&output_shape, new_data)
}

// returns the maximum of the entire tensor or an axis of the tensor
pub fn max<B: Backend>(tensor: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, TensorError> {
    let output_shape = reduced_shape(tensor, axis)?;
    let new_data = B::max(tensor.storage(), tensor.shape(), axis);
    Tensor::from_storage(&output_shape, new_data)
}

// returns the minimum of the entire tensor or an axis of the tensor
pub fn min<B: Backend>(tensor: &Tensor<B>, axis: Option<usize>) -> Result<Tensor<B>, TensorError> {
    let output_shape = reduced_shape(tensor, axis)?;
    let new_data = B::min(tensor.storage(), tensor.shape(), axis);
    Tensor::from_storage(&output_shape, new_data)
}
