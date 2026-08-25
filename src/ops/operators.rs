use crate::{backend::Backend, tensor::Tensor};

impl<B: Backend> std::ops::Add for Tensor<B> {
    type Output = Tensor<B>;

    fn add(self, rhs: Tensor<B>) -> Tensor<B> {
        crate::ops::add(&self, &rhs).expect("Tensor addition failed")
    }
}

impl<B: Backend> std::ops::Sub for Tensor<B> {
    type Output = Tensor<B>;

    fn sub(self, rhs: Tensor<B>) -> Tensor<B> {
        crate::ops::sub(&self, &rhs).expect("Tensor subtraction failed")
    }
}

impl<B: Backend> std::ops::Mul for Tensor<B> {
    type Output = Tensor<B>;

    fn mul(self, rhs: Tensor<B>) -> Tensor<B> {
        crate::ops::mult(&self, &rhs).expect("Tensor multiplication failed")
    }
}

impl<B: Backend> std::ops::Div for Tensor<B> {
    type Output = Tensor<B>;

    fn div(self, rhs: Tensor<B>) -> Tensor<B> {
        crate::ops::div(&self, &rhs).expect("Tensor division failed")
    }
}

impl<B: Backend> std::ops::Neg for Tensor<B> {
    type Output = Tensor<B>;

    fn neg(self) -> Tensor<B> {
        crate::ops::neg(&self).expect("Tensor negation failed")
    }
}
