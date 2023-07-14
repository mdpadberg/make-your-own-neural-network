use anyhow::{ensure, Context, Ok, Result};
use rand::Rng;

#[derive(Debug, PartialEq, Clone)]
pub struct Matrix(Vec<Vec<f64>>);

impl Matrix {
    pub fn new(data: Vec<Vec<f64>>) -> Matrix {
        Matrix(data)
    }

    pub fn get_data(self) -> Vec<Vec<f64>> {
        self.0
    }

    pub fn get_data_as_ref(&self) -> &Vec<Vec<f64>> {
        &self.0
    }

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
        Matrix(values)
    }

    /// This function multiplies a float to the all the values in the matrix
    pub fn multiply_value_to_all_values_in_matrix(&mut self, value: &f64) {
        for row in self.0.iter_mut() {
            for current in row.iter_mut() {
                *current *= value
            }
        }
    }

    /// This function adds a matrix to another matrix of the same size
    pub fn add_matrix_of_same_size(&mut self, matrix: &Matrix) -> Result<()> {
        ensure!(
            self.0.len() == matrix.get_data_as_ref().len(),
            "Matrix add: the row size of both matrices should be the same"
        );
        for (i, row) in self.0.iter_mut().enumerate() {
            ensure!(
                row.len()
                    == matrix
                        .get_data_as_ref()
                        .get(i)
                        .context("Matrix add: cannot retrieve row")?
                        .len(),
                "Matrix add: the column size of both matrices should be the same"
            );
            for (j, current) in row.iter_mut().enumerate() {
                *current += matrix.get_data_as_ref()[i][j]
            }
        }
        Ok(())
    }

    /// This function subtracts a matrix to another matrix of the same size
    pub fn subtract_matrix_of_same_size(&mut self, matrix: &Matrix) -> Result<()> {
        ensure!(
            self.0.len() == matrix.get_data_as_ref().len(),
            "Matrix subtract: the row size of both matrices should be the same"
        );
        for (i, row) in self.0.iter_mut().enumerate() {
            ensure!(
                row.len()
                    == matrix
                        .get_data_as_ref()
                        .get(i)
                        .context("Matrix subtract: cannot retrieve row")?
                        .len(),
                "Matrix subtract: the column size of both matrices should be the same"
            );
            for (j, current) in row.iter_mut().enumerate() {
                *current -= matrix.get_data_as_ref()[i][j]
            }
        }
        Ok(())
    }

    pub fn multiply(&self, matrix: &Matrix) -> Result<Matrix> {
        if self.0.len() == matrix.get_data_as_ref().len() 
        && self.0[0].len() == matrix.get_data_as_ref()[0].len() {
            self.multiply_matrix_of_same_size(matrix)
        } else {
            self.matrix_multiplication(matrix)
        }
    }

    /// This function subtracts a matrix to another matrix of the same size
    fn multiply_matrix_of_same_size(&self, matrix: &Matrix) -> Result<Matrix> {
        ensure!(
            self.0.len() == matrix.get_data_as_ref().len(),
            "Matrix multiply: the row size of both matrices should be the same"
        );
        let (self_rows, self_cols) = matrix_rows_and_cols(self);
        let mut new_matrix = Matrix(vec![vec![0.0; self_cols]; self_rows]);
        for (i, row) in self.0.iter().enumerate() {
            ensure!(
                row.len()
                    == matrix
                        .get_data_as_ref()
                        .get(i)
                        .context("Matrix multiply: cannot retrieve row")?
                        .len(),
                "Matrix multiply: the column size of both matrices should be the same"
            );
            for (j, current) in row.iter().enumerate() {
                new_matrix.0[i][j] = *current * matrix.get_data_as_ref()[i][j]
            }
        }
        Ok(new_matrix)
    }

    /// Matrix Multiplication:
    /// In mathematics, particularly in linear algebra, matrix multiplication is a binary operation that produces a matrix from two matrices.
    /// For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix.
    /// The resulting matrix, known as the matrix product, has the number of rows of the first and the number of columns of the second matrix.
    /// The product of matrices A and B is denoted as AB.
    fn matrix_multiplication(&self, matrix: &Matrix) -> Result<Matrix> {
        let (self_rows, self_cols) = matrix_rows_and_cols(self);
        let (matrix_rows, matrix_cols) = matrix_rows_and_cols(&matrix);
        ensure!(
            self_cols == matrix_rows,
            "Matrix: the number of columns in the first matrix must be equal to the number of rows in the second matrix"
        );
        let mut new_matrix = Matrix(vec![vec![0.0; matrix_cols]; self_rows]);
        for i in 0..self_rows {
            for j in 0..matrix_cols {
                let mut k = 0;
                loop {
                    if k < self_cols && k < matrix_rows {
                        new_matrix.0[i][j] += self.0[i][k] * matrix.0[k][j];
                    } else {
                        break;
                    }
                    k += 1;
                }
            }
        }
        Ok(new_matrix)
    }

    pub fn apply_activation_function(&self) -> Matrix {
        self.apply_sigmoid()
    }

    /// Sigmoid function:
    /// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    /// Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often
    /// from 0 to 1 or alternatively from −1 to 1, depending on convention.
    fn apply_sigmoid(&self) -> Matrix {
        let (rows, cols) = matrix_rows_and_cols(&self);
        let mut rows = Vec::with_capacity(rows);
        for row in self.0.iter() {
            let mut cols = Vec::with_capacity(cols);
            for value in row.iter() {
                cols.push(sigmoid(*value));
            }
            rows.push(cols);
        }
        Matrix(rows)
    }

    pub fn one_minus_all_values(self) -> Matrix {
        let (rows, cols) = matrix_rows_and_cols(&self);
        let mut rows = Vec::with_capacity(rows);
        for row in self.0.into_iter() {
            let mut cols = Vec::with_capacity(cols);
            for value in row.into_iter() {
                cols.push(1.0 - value);
            }
            rows.push(cols);
        }
        Matrix(rows)
    }

    /// Transpose:
    /// In linear algebra, the transpose of a matrix is an operator which flips a matrix over its diagonal,
    /// that is it switches the row and column indices of the matrix by producing another matrix denoted
    /// as AT (also written A′, Atr, tA or At).
    pub fn transpose(&self) -> Matrix {
        let (self_rows, self_cols) = matrix_rows_and_cols(self);
        let mut new_matrix = Matrix(vec![vec![0.0; self_rows]; self_cols]);
        for y in 0..self_rows {
            for x in 0..self_cols {
                new_matrix.0[x][y] = self.0[y][x];
            }
        }
        new_matrix
    }
}

fn sigmoid(input: f64) -> f64 {
    /// Euler's number (e)
    let e: f64 = std::f64::consts::E;
    1.0 / (1.0 + e.powf(-input))
}

fn matrix_rows_and_cols(matrix: &Matrix) -> (usize, usize) {
    let rows = matrix.0.len();
    let cols = if matrix.0.len() <= 0 {
        0
    } else {
        matrix.0[0].len()
    };
    (rows, cols)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn create_matrix() {
        let matrix = Matrix::new_with_random_values(3, 3);
        assert_eq!(matrix.0.len(), 3);
        assert_eq!(matrix.0[0].len(), 3);
    }

    #[test]
    fn add_value_to_matrix() {
        let mut matrix_one = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let matrix_two = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let result = matrix_one.add_matrix_of_same_size(&matrix_two);
        assert_eq!(result.is_ok(), true);
        assert_eq!(
            matrix_one,
            Matrix(vec![
                vec![0.2, 0.4, 0.6],
                vec![0.8, 1.0, 1.2],
                vec![1.4, 1.6, 1.8]
            ])
        );
    }

    #[test]
    fn add_value_to_matrix_2() {
        let mut matrix_one = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let matrix_two = Matrix(vec![vec![0.1, 0.2], vec![0.4, 0.5], vec![0.7, 0.8]]);
        let result = matrix_one.add_matrix_of_same_size(&matrix_two);
        assert_eq!(result.is_err(), true);
        assert_eq!(
            matrix_one,
            Matrix(vec![
                vec![0.1, 0.2, 0.3],
                vec![0.4, 0.5, 0.6],
                vec![0.7, 0.8, 0.9]
            ])
        );
    }

    #[test]
    fn substract_value_to_matrix() {
        let mut matrix_one = Matrix(vec![
            vec![1.5, 1.5, 1.5],
            vec![1.5, 1.5, 1.5],
            vec![1.5, 1.5, 1.5],
        ]);
        let matrix_two = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let result = matrix_one.subtract_matrix_of_same_size(&matrix_two);
        assert_eq!(result.is_ok(), true);
        assert_eq!(
            matrix_one,
            Matrix(vec![
                vec![1.4, 1.3, 1.2],
                vec![1.1, 1.0, 0.9],
                vec![0.8, 0.7, 0.6]
            ])
        );
    }

    #[test]
    fn substract_value_to_matrix_2() {
        let mut matrix_one = Matrix(vec![
            vec![1.5, 1.5, 1.5],
            vec![1.5, 1.5, 1.5],
            vec![1.5, 1.5, 1.5],
        ]);
        let matrix_two = Matrix(vec![vec![0.1, 0.2], vec![0.4, 0.5], vec![0.7, 0.8]]);
        let result = matrix_one.subtract_matrix_of_same_size(&matrix_two);
        assert_eq!(result.is_err(), true);
        assert_eq!(
            matrix_one,
            Matrix(vec![
                vec![1.5, 1.5, 1.5],
                vec![1.5, 1.5, 1.5],
                vec![1.5, 1.5, 1.5],
            ])
        );
    }

    #[test]
    fn testing_matrixMultiplication_1() {
        let matrix_one = Matrix(vec![
            vec![1.0, 0.0, 1.0],
            vec![2.0, 1.0, 1.0],
            vec![0.0, 1.0, 1.0],
            vec![1.0, 1.0, 2.0],
        ]);
        let matrix_two = Matrix(vec![
            vec![1.0, 2.0, 1.0],
            vec![2.0, 3.0, 1.0],
            vec![4.0, 2.0, 2.0],
        ]);
        assert_eq!(
            matrix_one.matrix_multiplication(&matrix_two).unwrap(),
            Matrix(vec![
                vec![5.0, 4.0, 3.0],
                vec![8.0, 9.0, 5.0],
                vec![6.0, 5.0, 3.0],
                vec![11.0, 9.0, 6.0],
            ])
        );
    }

    #[test]
    fn testing_transpose_1() {
        let matrix_one = Matrix(vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]]);
        assert_eq!(
            matrix_one.transpose(),
            Matrix(vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]])
        );
    }

    #[test]
    fn testing_sigmoid() {
        assert_eq!(sigmoid(3.0), 0.9525741268224331);
    }
}
