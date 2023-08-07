use anyhow::{ensure, Context, Result};

use crate::matrix::matrix::Matrix;
use crate::neuralnetwork::layer::Layer;
use crate::neuralnetwork::neuralnetwork::NeuralNetwork;

#[derive(Debug, PartialEq)]
pub(crate) struct Feedforward {
    pub(crate) results: Vec<Matrix>,
}

impl Feedforward {
    pub(crate) fn run(neural_network: NeuralNetwork, input_data: &Vec<f64>) -> Result<Feedforward> {
        ensure!(
            neural_network.amount_of_input_neurons == (input_data.len() as u32),
            "Feedforward: The input data should have the same size as the amount of input neurons"
        );
        let feedforward = calculate_results_per_layer(
            &neural_network.layers,
            transform_input_data_to_matrix(input_data),
        )?;
        ensure!((feedforward.results
            .get(1)
            .context("Feedforward: No data in hidden layer")?
            .0
            .len() as u32) == neural_network.amount_of_hidden_neurons,
            "Feedforward: Result of hidden layer should be of same size as amount of hidden neurons"
        );
        ensure!((feedforward.results
            .last()
            .context("Feedforward: no last layer")?
            .0
            .len() as u32) == neural_network.amount_of_output_neurons,
            "Feedforward: Result of output layer should be of same size as amount of output neurons"
        );
        Ok(feedforward)
    }
}

/// calculate_results_per_layer:
/// This method returns the result of each layer. Starting at the input data and then going
/// through the whole neural network. This way you can see how much each layer contributed to
/// the end result of the neural network
fn calculate_results_per_layer(layers: &Vec<Layer>, input_data: Matrix) -> Result<Feedforward> {
    let mut result: Vec<Matrix> = vec![input_data.clone()];
    for layer in layers {
        let result_from_last_layer = if result.is_empty() {
            &input_data
        } else {
            result.last().context("Feedforward: No last layer")?
        };
        let result_from_current_layer: Matrix =
            (&layer.0 * result_from_last_layer)?.apply_sigmoid();
        result.push(result_from_current_layer);
    }
    Ok(Feedforward { results: result })
}

fn transform_input_data_to_matrix(input_data: &Vec<f64>) -> Matrix {
    Matrix(
        input_data
            .into_iter()
            .map(|value| vec![*value])
            .collect::<Vec<Vec<f64>>>(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn feedforward_example_from_the_book() {
        let input = vec![0.9, 0.1, 0.8];
        let actual_result = calculate_results_per_layer(
            &vec![
                // input to hidden weights
                Layer(Matrix(vec![
                    vec![0.9, 0.3, 0.4],
                    vec![0.2, 0.8, 0.2],
                    vec![0.1, 0.5, 0.6],
                ])),
                // hidden to output weights
                Layer(Matrix(vec![
                    vec![0.3, 0.7, 0.5],
                    vec![0.6, 0.5, 0.2],
                    vec![0.8, 0.1, 0.9],
                ])),
            ],
            transform_input_data_to_matrix(&input),
        )
        .unwrap();
        let expected_result = vec![
            Matrix(vec![vec![0.9], vec![0.1], vec![0.8]]),
            Matrix(vec![
                vec![0.7613327148429104],
                vec![0.6034832498647263],
                vec![0.6502185485738271],
            ]),
            Matrix(vec![
                vec![0.7263033450139793],
                vec![0.7085980724248232],
                vec![0.778097059561142],
            ]),
        ];
        assert_eq!(actual_result.results, expected_result);
    }

    #[test]
    fn feedforward_my_own_example_with_multiple_hidden_layers_and_with_different_sizes() {
        let input = vec![0.1, 0.2, 0.3];
        let actual_result = calculate_results_per_layer(
            &vec![
                // input to hidden weights
                Layer(Matrix(vec![
                    vec![1.1, 2.1, 3.1],
                    vec![1.2, 2.2, 3.2],
                    vec![1.3, 2.3, 3.3],
                    vec![1.4, 2.4, 3.4],
                ])),
                // hidden to hidden weights
                Layer(Matrix(vec![
                    vec![1.1, 2.1, 3.1, 4.1],
                    vec![1.2, 2.2, 3.2, 4.2],
                    vec![1.3, 2.3, 3.3, 4.3],
                    vec![1.4, 2.4, 3.4, 4.4],
                ])),
                // hidden to output weights
                Layer(Matrix(vec![
                    vec![1.1, 2.1, 3.1, 4.1],
                    vec![1.2, 2.2, 3.2, 4.2],
                    vec![1.3, 2.3, 3.3, 4.3],
                ])),
            ],
            transform_input_data_to_matrix(&input),
        )
        .unwrap();
        let expected_result = vec![
            Matrix(vec![vec![0.1], vec![0.2], vec![0.3]]),
            Matrix(vec![
                vec![0.8115326747861805],
                vec![0.8205384805926733],
                vec![0.8292045179776254],
                vec![0.8375349374193038],
            ]),
            Matrix(vec![
                vec![0.9998196163164922],
                vec![0.9998702958894329],
                vec![0.9999067381461998],
                vec![0.9999329421074687],
            ]),
            Matrix(vec![
                vec![0.9999695369356685],
                vec![0.9999795788316486],
                vec![0.9999863105453931],
            ]),
        ];
        assert_eq!(actual_result.results, expected_result);
    }

    #[test]
    fn test_input_ensure() {
        let result = Feedforward::run(NeuralNetwork::new(3, 3, 3, 1), &vec![0.0]);
        assert_eq!(result.is_err(), true);
        let error = result.unwrap_err();
        let mut chain = error.chain();
        assert_eq!(
            chain.next().map(|x| format!("{x}")),
            Some("Feedforward: The input data should have the same size as the amount of input neurons".to_owned())
        );
        assert_eq!(chain.next().map(|x| format!("{x}")), None);
    }
}
