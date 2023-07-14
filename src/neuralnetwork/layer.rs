use crate::matrix::matrix::Matrix;

#[derive(Debug, Clone, PartialEq)]
pub struct Layer(Matrix);

impl Layer {
    pub fn new(data: Matrix) -> Layer {
        Layer(data)
    }

    pub fn get_matrix_as_ref(&self) -> &Matrix {
        &self.0
    }

    pub fn get_matrix(self) -> Matrix {
        self.0
    }
}