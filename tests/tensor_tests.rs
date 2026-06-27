use chora::{Tensor, TensorShape};

#[test]
fn test1() {
    let shape = TensorShape::new(vec![2, 3]);
    let mut tensor = Tensor::new(shape, vec![0, 1, 2, 3, 4, 5]);
    tensor[&[1, 2]] = 8;
    assert_eq!(tensor[&[1, 2]], 8);
}
#[test]
fn test_print_1d() {
    let shape = TensorShape::new(vec![6]);
    let tensor = Tensor::new(shape, vec![0, 1, 2, 3, 4, 5]);
    assert_eq!(tensor.to_string(), "[0, 1, 2, 3, 4, 5]");
}
