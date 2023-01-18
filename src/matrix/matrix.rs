use anyhow::{bail, Ok, Result};
use rand::Rng;

#[derive(Debug, PartialEq)]
pub struct Matrix {
    values: Vec<Vec<f64>>,
}

impl Matrix {
    /// This function creates a matrix with an y amount of rows and an x amount of cols
    pub fn new_with_random_values(amount_of_rows: u32, amount_of_cols: u32) -> Matrix {
        let mut rng = rand::thread_rng();
        let mut values: Vec<Vec<f64>> = vec![];
        for _ in 0..amount_of_rows {
            let mut row: Vec<f64> = vec![];
            for _ in 0..amount_of_cols {
                row.push(rng.gen());
            }
            values.push(row);
        }
        Matrix { values }
    }

    /// This function adds a float to the all the values in the matrix
    pub fn add_value_to_all_values_in_matrix(&mut self, value: f64) {
        for row in self.values.iter_mut() {
            for current in row.iter_mut() {
                *current += value
            }
        }
    }

    /// This function subtracts a float to the all the values in the matrix
    pub fn subtract_value_to_all_values_in_matrix(&mut self, value: f64) {
        for row in self.values.iter_mut() {
            for current in row.iter_mut() {
                *current -= value
            }
        }
    }

    /// This function multiplies a float to the all the values in the matrix
    pub fn multiply_value_to_all_values_in_matrix(&mut self, value: f64) {
        for row in self.values.iter_mut() {
            for current in row.iter_mut() {
                *current *= value
            }
        }
    }

    /// Sigmoid function:
    /// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    /// Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often
    /// from 0 to 1 or alternatively from −1 to 1, depending on convention.
    pub fn sigmoid(&self) -> Result<Matrix> {
        let mut new_matrix = Matrix {values: vec![]};
        for row in self.values.iter() {
            let mut new_row = vec![];
            for value in row.iter() {
                new_row.push(sigmoid(value));
            }
            new_matrix.values.push(new_row);
        }
        Ok(new_matrix)
    }

    /// Derivative of Sigmoid function = sigmoid(input) * (1 - sigmoid(input))
    pub fn derivative_of_sigmoid(&self) -> Result<Matrix> {
        let mut new_matrix = Matrix {values: vec![]};
        for row in self.values.iter() {
            let mut new_row = vec![];
            for value in row.iter() {
                new_row.push(derivative_of_sigmoid(value));
            }
            new_matrix.values.push(new_row);
        }
        Ok(new_matrix)
    }

    /// Hadamard product (matrices):
    /// In mathematics, the Hadamard product (also known as the Schur product[1] or the entrywise product[2]) is a binary operation
    /// that takes two matrices of the same dimensions, and produces another matrix where each element i,j is the product of elements i,j
    /// of the original two matrices. It should not be confused with the more common matrix product. It is attributed to, and named after,
    /// either French mathematician Jacques Hadamard, or German mathematician Issai Schur.
    pub fn hadamard_product(&self, matrix: Matrix) -> Result<Matrix> {
        let rows_self = self.values.len();
        let rows_matrix = matrix.values.len();
        if
        // matrices are not from the same size
        rows_self != rows_matrix {
            bail!("matrices are not the same size");
        }
        let mut new_matrix = Matrix { values: vec![] };
        for (y, row) in self.values.iter().enumerate() {
            let mut new_row: Vec<f64> = vec![];
            for (x, current) in row.iter().enumerate() {
                new_row.push(current * matrix.values[y][x]);
            }
            new_matrix.values.push(new_row);
        }
        Ok(new_matrix)
    }

    /// Matrix Multiplication:
    /// In mathematics, particularly in linear algebra, matrix multiplication is a binary operation that produces a matrix from two matrices.
    /// For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix.
    /// The resulting matrix, known as the matrix product, has the number of rows of the first and the number of columns of the second matrix.
    /// The product of matrices A and B is denoted as AB.
    pub fn matrix_multiplication(&self, matrix: Matrix) -> Result<Matrix> {
        let self_rows = self.values.len();
        let self_cols = if self.values.len() <= 0 {
            0
        } else {
            self.values[0].len()
        };
        let matrix_rows = matrix.values.len();
        let matrix_cols = if matrix.values.len() <= 0 {
            0
        } else {
            matrix.values[0].len()
        };
        let mut new_matrix = Matrix {
            values: vec![vec![0.0; matrix_cols]; self_rows],
        };
        for i in 0..self_rows {
            for j in 0..matrix_cols {
                let mut k = 0;
                loop {
                    if k < self_cols && k < matrix_rows {
                        new_matrix.values[i][j] += self.values[i][k] * matrix.values[k][j];
                    } else {
                        break;
                    }
                    k += 1;
                }
            }
        }
        Ok(new_matrix)
    }

    /// Transpose:
    /// In linear algebra, the transpose of a matrix is an operator which flips a matrix over its diagonal,
    /// that is it switches the row and column indices of the matrix by producing another matrix denoted
    /// as AT (also written A′, Atr, tA or At).
    pub fn transpose(&self) -> Result<Matrix> {
        let self_rows = self.values.len();
        let self_cols = if self.values.len() <= 0 {
            0
        } else {
            self.values[0].len()
        };
        let mut new_matrix = Matrix {
            values: vec![vec![0.0; self_rows]; self_cols],
        };
        for y in 0..self_rows {
            for x in 0..self_cols {
                new_matrix.values[x][y] = self.values[y][x];
            }
        }
        Ok(new_matrix)
    }
}

/// Sigmoid function:
/// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
/// Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often
/// from 0 to 1 or alternatively from −1 to 1, depending on convention.
pub fn sigmoid(input: &f64) -> f64 {
    /// Euler's number (e)
    let e: f64 = std::f64::consts::E;
    1.0 / (1.0 + e.powf(-input))
}


/// Derivative of Sigmoid function = sigmoid(input) * (1 - sigmoid(input))
pub fn derivative_of_sigmoid(input: &f64) -> f64 {
    sigmoid(input) * (1.0 - sigmoid(input))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_matrix() {
        let matrix = Matrix::new_with_random_values(3, 3);
        assert_eq!(matrix.values.len(), 3);
        assert_eq!(matrix.values[0].len(), 3);
    }

    #[test]
    fn add_value_to_matrix() {
        let mut matrix = Matrix::new_with_random_values(3, 3);
        let original_values = matrix.values.clone();
        matrix.add_value_to_all_values_in_matrix(2.0);
        for (y, row) in matrix.values.iter().enumerate() {
            for (x, current) in row.iter().enumerate() {
                assert_eq!(&(original_values[y][x] + 2.0), current);
            }
        }
    }

    #[test]
    fn substract_value_to_matrix() {
        let mut matrix = Matrix::new_with_random_values(3, 3);
        let original_values = matrix.values.clone();
        matrix.subtract_value_to_all_values_in_matrix(2.0);
        for (y, row) in matrix.values.iter().enumerate() {
            for (x, current) in row.iter().enumerate() {
                assert_eq!(&(original_values[y][x] - 2.0), current);
            }
        }
    }

    #[test]
    fn multiply_value_to_matrix() {
        let mut matrix = Matrix::new_with_random_values(3, 3);
        let original_values = matrix.values.clone();
        matrix.multiply_value_to_all_values_in_matrix(2.0);
        for (y, row) in matrix.values.iter().enumerate() {
            for (x, current) in row.iter().enumerate() {
                assert_eq!(&(original_values[y][x] * 2.0), current);
            }
        }
    }

    #[test]
    fn testing_hadamardProduct_1() {
        let mut matrix_one = Matrix::new_with_random_values(3, 3);
        let mut matrix_two = Matrix::new_with_random_values(2, 2);
        assert!(matrix_one.hadamard_product(matrix_two).is_err());
    }

    #[test]
    fn testing_hadamardProduct_2() {
        let matrix_one = Matrix {
            values: vec![vec![3.0, 2.0], vec![0.0, 1.0], vec![-2.0, 5.0]],
        };
        let matrix_two = Matrix {
            values: vec![vec![1.0, 2.0], vec![3.0, 1.0], vec![3.0, 3.0]],
        };
        assert_eq!(
            matrix_one.hadamard_product(matrix_two).unwrap(),
            Matrix {
                values: vec![vec![3.0, 4.0], vec![0.0, 1.0], vec![-6.0, 15.0],],
            }
        );
    }

    #[test]
    fn testing_matrixMultiplication_1() {
        let matrix_one = Matrix {
            values: vec![
                vec![1.0, 0.0, 1.0],
                vec![2.0, 1.0, 1.0],
                vec![0.0, 1.0, 1.0],
                vec![1.0, 1.0, 2.0],
            ],
        };
        let matrix_two = Matrix {
            values: vec![
                vec![1.0, 2.0, 1.0],
                vec![2.0, 3.0, 1.0],
                vec![4.0, 2.0, 2.0],
            ],
        };
        assert_eq!(
            matrix_one.matrix_multiplication(matrix_two).unwrap(),
            Matrix {
                values: vec![
                    vec![5.0, 4.0, 3.0],
                    vec![8.0, 9.0, 5.0],
                    vec![6.0, 5.0, 3.0],
                    vec![11.0, 9.0, 6.0]
                ],
            }
        );
    }

    #[test]
    fn testing_transpose_1() {
        let matrix_one = Matrix {
            values: vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]],
        };
        assert_eq!(
            matrix_one.transpose().unwrap(),
            Matrix {
                values: vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0],],
            }
        );
    }

    #[test]
    fn testing_sigmoid() {
        assert_eq!(sigmoid(&3.0), 0.9525741268224331);
    }

    #[test]
    fn testing_derivative_of_sigmoid() {
        assert_eq!(derivative_of_sigmoid(&3.0), 0.0451766597309122);
    }
}
