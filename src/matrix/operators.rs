use super::matrix::{matrix_rows_and_cols, Matrix};
use anyhow::{ensure, Context, Result};
use std::fmt;
use std::ops::{Add, Mul, Sub};

#[derive(Debug)]
pub(crate) enum Operator {
    ADD,
    SUB,
    MUL,
}

impl fmt::Display for Operator {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Add<Matrix> for Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn add(self, rhs: Matrix) -> Self::Output {
        apply_operation_on_matrices_of_same_size(&self, &rhs, Operator::ADD)
    }
}

impl Add<&Matrix> for &Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn add(self, rhs: &Matrix) -> Self::Output {
        apply_operation_on_matrices_of_same_size(self, rhs, Operator::ADD)
    }
}

impl Sub<Matrix> for Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn sub(self, rhs: Matrix) -> Self::Output {
        apply_operation_on_matrices_of_same_size(&self, &rhs, Operator::SUB)
    }
}

impl Sub<&Matrix> for &Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn sub(self, rhs: &Matrix) -> Self::Output {
        apply_operation_on_matrices_of_same_size(self, rhs, Operator::SUB)
    }
}

impl Mul<Matrix> for Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn mul(self, rhs: Matrix) -> Self::Output {
        if self.0.len() == rhs.0.len() && self.0[0].len() == rhs.0[0].len() {
            apply_operation_on_matrices_of_same_size(&self, &rhs, Operator::MUL)
        } else {
            matrix_multiplication(&self, &rhs)
        }
    }
}

impl Mul<&Matrix> for Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.0.len() == rhs.0.len() && self.0[0].len() == rhs.0[0].len() {
            apply_operation_on_matrices_of_same_size(&self, rhs, Operator::MUL)
        } else {
            matrix_multiplication(&self, rhs)
        }
    }
}

impl Mul<&Matrix> for &Matrix {
    type Output = Result<Matrix, anyhow::Error>;

    fn mul(self, rhs: &Matrix) -> Self::Output {
        if self.0.len() == rhs.0.len() && self.0[0].len() == rhs.0[0].len() {
            apply_operation_on_matrices_of_same_size(self, rhs, Operator::MUL)
        } else {
            matrix_multiplication(self, rhs)
        }
    }
}

impl Mul<&f64> for Matrix {
    type Output = Matrix;

    fn mul(self, rhs: &f64) -> Self::Output {
        let (amount_of_rows, amount_of_cols) = matrix_rows_and_cols(&self);
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(amount_of_rows as usize);
        for i in 0..amount_of_rows {
            let mut cols: Vec<f64> = Vec::with_capacity(amount_of_cols as usize);
            for j in 0..amount_of_cols {
                cols.push(self.0[i][j] * rhs)
            }
            rows.push(cols);
        }
        Matrix(rows)
    }
}

fn apply_operation_on_matrices_of_same_size(
    matrix_one: &Matrix,
    matrix_two: &Matrix,
    operation: Operator,
) -> Result<Matrix> {
    ensure!(
        matrix_one.0.len() == matrix_two.0.len(),
        format!(
            "Matrix {}: the row size of both matrices should be the same",
            operation
        )
    );
    let mut rows: Vec<Vec<f64>> = Vec::with_capacity(matrix_one.0.len());
    for (i, row) in matrix_one.0.iter().enumerate() {
        ensure!(
            row.len()
                == matrix_two
                    .0
                    .get(i)
                    .context(format!("Matrix {}: cannot retrieve row", operation))?
                    .len(),
            format!(
                "Matrix {}: the column size of both matrices should be the same",
                operation
            )
        );
        let mut cols: Vec<f64> = Vec::with_capacity(row.len());
        for (j, current) in row.iter().enumerate() {
            match operation {
                Operator::ADD => cols.push(current + matrix_two.0[i][j]),
                Operator::SUB => cols.push(current - matrix_two.0[i][j]),
                Operator::MUL => cols.push(current * matrix_two.0[i][j]),
            }
        }
        rows.push(cols);
    }
    Ok(Matrix(rows))
}

/// Matrix Multiplication:
/// In mathematics, particularly in linear algebra, matrix multiplication is a binary operation that produces a matrix from two matrices.
/// For matrix multiplication, the number of columns in the first matrix must be equal to the number of rows in the second matrix.
/// The resulting matrix, known as the matrix product, has the number of rows of the first and the number of columns of the second matrix.
/// The product of matrices A and B is denoted as AB.
fn matrix_multiplication(matrix_one: &Matrix, matrix_two: &Matrix) -> Result<Matrix> {
    let (matrix_one_rows, matrix_one_cols) = matrix_rows_and_cols(&matrix_one);
    let (matrix_two_rows, matrix_two_cols) = matrix_rows_and_cols(&matrix_two);
    ensure!(
            matrix_one_cols == matrix_two_rows,
            "Matrix: the number of columns in the first matrix must be equal to the number of rows in the second matrix"
        );
    let mut new_matrix = Matrix(vec![vec![0.0; matrix_two_cols]; matrix_one_rows]);
    for i in 0..matrix_one_rows {
        for j in 0..matrix_two_cols {
            let mut k = 0;
            loop {
                if k < matrix_one_cols && k < matrix_two_rows {
                    new_matrix.0[i][j] += matrix_one.0[i][k] * matrix_two.0[k][j];
                } else {
                    break;
                }
                k += 1;
            }
        }
    }
    Ok(new_matrix)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn testing_add() {
        let matrix_one = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let matrix_two = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        assert_eq!(
            &(&matrix_one + &matrix_two).unwrap(),
            &Matrix(vec![
                vec![0.2, 0.4, 0.6],
                vec![0.8, 1.0, 1.2],
                vec![1.4, 1.6, 1.8],
            ])
        );
        assert_eq!(
            (matrix_one + matrix_two).unwrap(),
            Matrix(vec![
                vec![0.2, 0.4, 0.6],
                vec![0.8, 1.0, 1.2],
                vec![1.4, 1.6, 1.8],
            ])
        );
    }

    #[test]
    fn testing_sub() {
        let matrix_one = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let matrix_two = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        assert_eq!(
            &(&matrix_one - &matrix_two).unwrap(),
            &Matrix(vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ])
        );
        assert_eq!(
            (matrix_one - matrix_two).unwrap(),
            Matrix(vec![
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
                vec![0.0, 0.0, 0.0],
            ])
        );
    }

    #[test]
    fn testing_mul() {
        let matrix_one = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        let matrix_two = Matrix(vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6],
            vec![0.7, 0.8, 0.9],
        ]);
        assert_eq!(
            &(&matrix_one * &matrix_two).unwrap(),
            &Matrix(vec![
                vec![0.010000000000000002, 0.04000000000000001, 0.09],
                vec![0.16000000000000003, 0.25, 0.36],
                vec![0.48999999999999994, 0.6400000000000001, 0.81],
            ])
        );
        assert_eq!(
            (matrix_one * matrix_two).unwrap(),
            Matrix(vec![
                vec![0.010000000000000002, 0.04000000000000001, 0.09],
                vec![0.16000000000000003, 0.25, 0.36],
                vec![0.48999999999999994, 0.6400000000000001, 0.81],
            ])
        );
    }

    #[test]
    fn testing_matrix_multiplication() {
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
            (matrix_one * matrix_two).unwrap(),
            Matrix(vec![
                vec![5.0, 4.0, 3.0],
                vec![8.0, 9.0, 5.0],
                vec![6.0, 5.0, 3.0],
                vec![11.0, 9.0, 6.0],
            ])
        );
    }
}
