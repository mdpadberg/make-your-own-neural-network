use crate::logic::feedforward::Feedforward;
use crate::matrix::matrix::Matrix;
use crate::neuralnetwork::layer::Layer;
use anyhow::{ensure, Context, Result};

#[derive(Debug)]
pub struct Backpropagation {}

impl Backpropagation {
    pub fn run(target: &Vec<f64>, layers: &Vec<Layer>, feedforward: &Feedforward) {
        //ensure checks
        //target same size as output layer, check amount of layers with feedforward size etc
    }
}

/// calculate_error_rate_per_layer:
/// This method returns the error rate per layer, by calculating the error rate from the
/// output layer (target - actual). Then going through each layer from the back to the front,
/// while skipping the input layer. This way you can see how much each layer contributed to
/// the error rate of the neural network
fn calculate_error_rate_per_layer(
    target: Matrix,
    actual: &Matrix,
    layers: &Vec<Layer>,
) -> Result<Vec<Matrix>> {
    let mut result: Vec<Matrix> = vec![error_rate_from_last_layer(target, actual)?];
    for layer in layers.iter().skip(1).rev() {
        let result_from_last_processed_layer =
            result.last().context("Backpropagation: No last layer")?;
        let current_matrix = layer.get_matrix_as_ref().transpose();
        result.push(current_matrix.matrix_multiplication(result_from_last_processed_layer)?);
    }
    result.reverse();
    Ok(result)
}

fn transform_target_data_to_matrix(target_data: &Vec<f64>) -> Matrix {
    Matrix::new(
        target_data
            .into_iter()
            .map(|value| vec![*value])
            .collect::<Vec<Vec<f64>>>(),
    )
}

fn error_rate_from_last_layer(target: Matrix, actual: &Matrix) -> Result<Matrix> {
    let mut error_rate_from_last_layer = target;
    error_rate_from_last_layer.subtract_matrix_of_same_size(actual)?;
    Ok(error_rate_from_last_layer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn calculate_error_rate_example_from_the_book() {
        let layers = vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![vec![0.42, 0.28], vec![0.16, 1.1375]])),
            Layer::new(Matrix::new(vec![vec![0.4333333333333333,0.7333333333333334], vec![0.1,0.4]])),
        ];
        let actual = calculate_error_rate_per_layer(
            Matrix::new(vec![vec![3.0], vec![3.0]]),
            &Matrix::new(vec![vec![1.5], vec![2.5]]),
            &layers,
        )
        .unwrap();
        let expected = vec![
            Matrix::new(vec![
                vec![0.7],
                vec![1.3],
            ]),
            Matrix::new(vec![
                vec![1.5],
                vec![0.5],
            ]),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculate_error_rate_example_with_one_hidden_layers_and_with_different_sizes() {
        let layers = vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![
                vec![-0.62091563, -1.25154723, -0.14334102],
                vec![0.07702094, -0.04681276, 0.62532329],
                vec![0.64765397, -0.13824144, 0.04829786],
                vec![0.58148507, -0.63079071, -0.38690071],
            ])),
            // hidden to output weights
            Layer::new(Matrix::new(vec![
                vec![-0.12369625, 0.16732224, -0.02174357, -0.06230827],
                vec![0.32413962, 0.2632736, -0.01882388, 0.12343084],
                vec![0.23642887, -0.44747458, 0.14426986, -0.11192151],
            ])),
        ];
        let actual = calculate_error_rate_per_layer(
            Matrix::new(vec![vec![1.0], vec![0.0], vec![0.0]]),
            &Matrix::new(vec![vec![0.50024943], vec![0.58025132], vec![0.46906931]]),
            &layers,
        )
        .unwrap();
        let expected = vec![
            Matrix::new(vec![
                vec![-0.3608012407286406],
                vec![0.14075112337566462],
                vec![-0.0676163439678099],
                vec![-0.05026055581706461],
            ]),
            Matrix::new(vec![vec![0.49975057], vec![-0.58025132], vec![-0.46906931]]),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculate_error_rate_example_with_multiple_hidden_layers_and_with_different_sizes() {
        let layers = &vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1],
                vec![1.2, 2.2, 3.2],
                vec![1.3, 2.3, 3.3],
                vec![1.4, 2.4, 3.4],
            ])),
            // hidden to hidden weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
                vec![1.4, 2.4, 3.4, 4.4],
            ])),
            // hidden to output weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
            ])),
        ];
        let actual = calculate_error_rate_per_layer(
            Matrix::new(vec![vec![2.0], vec![3.0], vec![6.0]]),
            &Matrix::new(vec![
                vec![0.9999695369356685],
                vec![0.9999795788316486],
                vec![0.9999863105453931],
            ]),
            &layers,
        )
        .unwrap();
        let expected = vec![
            Matrix::new(vec![
                vec![114.0008956448172],
                vec![202.00158633119605],
                vec![290.0022770175749],
                vec![378.0029677039537],
            ]),
            Matrix::new(vec![
                vec![10.000075811063777],
                vec![18.000140384751067],
                vec![26.000204958438356],
                vec![34.00026953212564],
            ]),
            Matrix::new(vec![
                vec![1.0000304630643315],
                vec![2.0000204211683514],
                vec![5.000013689454607],
            ]),
        ];
        assert_eq!(actual, expected);
    }
}
