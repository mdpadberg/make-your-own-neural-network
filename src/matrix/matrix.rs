use rand::Rng;

#[derive(Debug, PartialEq, Clone)]

pub(crate) struct Matrix(pub(crate) Vec<Vec<f64>>);

impl Matrix {
    /// This function creates a matrix with random values an y amount of rows and an x amount of cols
    pub(crate) fn new_with_random_values(amount_of_rows: u32, amount_of_cols: u32) -> Matrix {
        let mut rng = rand::thread_rng();
        let mut rows: Vec<Vec<f64>> = Vec::with_capacity(amount_of_rows as usize);
        for _ in 0..amount_of_rows {
            let mut cols: Vec<f64> = Vec::with_capacity(amount_of_cols as usize);
            for _ in 0..amount_of_cols {
                cols.push(rng.gen());
            }
            rows.push(cols);
        }
        Matrix(rows)
    }

    /// Sigmoid function:
    /// A sigmoid function is a mathematical function having a characteristic "S"-shaped curve or sigmoid curve.
    /// Sigmoid functions have domain of all real numbers, with return value monotonically increasing most often
    /// from 0 to 1 or alternatively from −1 to 1, depending on convention.
    pub(crate) fn apply_sigmoid(self) -> Matrix {
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

    pub(crate) fn one_minus_all_values(self) -> Matrix {
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
    pub(crate) fn transpose(&self) -> Matrix {
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

pub(crate) fn matrix_rows_and_cols(matrix: &Matrix) -> (usize, usize) {
    let rows = matrix.0.len();
    let cols = if matrix.0.len() <= 0 {
        0
    } else {
        matrix.0[0].len()
    };
    (rows, cols)
}

fn sigmoid(input: f64) -> f64 {
    /// Euler's number (e)
    let e: f64 = std::f64::consts::E;
    1.0 / (1.0 + e.powf(-input))
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
    fn testing_transpose() {
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

    #[test]
    fn testing_matrix_rows_and_cols() {
        let (rows, cols) = matrix_rows_and_cols(&Matrix(vec![vec![1.0, 3.0, 5.0], vec![2.0, 4.0, 6.0]]));
        assert_eq!(rows, 2);
        assert_eq!(cols, 3);
    }
}