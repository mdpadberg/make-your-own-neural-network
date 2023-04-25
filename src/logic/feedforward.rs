use anyhow::{Context, ensure, Result};

use crate::matrix::matrix::Matrix;
use crate::neuralnetwork::layer::Layer;
use crate::neuralnetwork::neuralnetwork::NeuralNetwork;

#[derive(Debug, PartialEq)]
pub struct Feedforward {
    pub results: Vec<Matrix>,
}

impl Feedforward {
    pub fn run(neural_network: NeuralNetwork, input_data: &Vec<f64>) -> Result<Feedforward> {
        ensure!(
            neural_network.amount_of_input_neurons() == &(input_data.len() as u32),
            "Feedforward: The input data should have the same size as the amount of input neurons"
        );
        let feedforward = run(
            neural_network.layers(),
            transform_input_data_to_matrix(input_data),
        )?;
        ensure!(&(feedforward.results
            .get(1)
            .context("Feedforward: No data in hidden layer")?
            .get_data_as_ref()
            .len() as u32) == neural_network.amount_of_hidden_neurons(),
            "Feedforward: Result of hidden layer should be of same size as amount of hidden neurons"
        );
        ensure!(&(feedforward.results
            .last()
            .context("Feedforward: no last layer")?
            .get_data_as_ref()
            .len() as u32) == neural_network.amount_of_output_neurons(),
            "Feedforward: Result of output layer should be of same size as amount of output neurons"
        );
        Ok(Feedforward { results: vec![] })
    }
}

fn run(layers: &Vec<Layer>, input_data: Matrix) -> Result<Feedforward> {
    let mut result: Vec<Matrix> = vec![input_data];
    for layer in layers {
        let result_from_last_layer = result.last().context("Feedforward: No last layer")?;
        let mut result_from_current_layer: Matrix = layer
            .get_matrix_as_ref()
            .matrix_multiplication(result_from_last_layer)?;
        result_from_current_layer.apply_activation_function();
        result.push(result_from_current_layer);
    }
    Ok(Feedforward { results: result })
}

fn transform_input_data_to_matrix(input_data: &Vec<f64>) -> Matrix {
    Matrix::new(
        input_data
            .into_iter()
            .map(|value| vec![*value])
            .collect::<Vec<Vec<f64>>>(),
    )
}

#[cfg(test)]
mod tests {
    use anyhow::{anyhow, Error};

    use crate::logic::feedforward;

    use super::*;

    #[test]
    fn feedforward_example_from_the_book() {
        let input = vec![0.9, 0.1, 0.8];
        let actual_result = run(
            &vec![
                // input to hidden weights
                Layer::new(Matrix::new(vec![
                    vec![0.9, 0.3, 0.4],
                    vec![0.2, 0.8, 0.2],
                    vec![0.1, 0.5, 0.6],
                ])),
                // hidden to output weights
                Layer::new(Matrix::new(vec![
                    vec![0.3, 0.7, 0.5],
                    vec![0.6, 0.5, 0.2],
                    vec![0.8, 0.1, 0.9],
                ])),
            ],
            transform_input_data_to_matrix(&input),
        )
        .unwrap();
        let expected_result = vec![
            Matrix::new(vec![vec![0.9], vec![0.1], vec![0.8]]),
            Matrix::new(vec![
                vec![0.7613327148429104],
                vec![0.6034832498647263],
                vec![0.6502185485738271],
            ]),
            Matrix::new(vec![
                vec![0.7263033450139793],
                vec![0.7085980724248232],
                vec![0.778097059561142],
            ]),
        ];
        assert_eq!(actual_result.results, expected_result);
    }

    #[test]
    fn feedforward_my_own_example_with_multiple_hidden_layers_and_with_different_sizes() {
        let input = vec![1.0, 0.5, -1.5];
        let actual_result = run(
            &vec![
                // input to hidden weights
                Layer::new(Matrix::new(vec![
                    vec![-1.03531205, 0.29141863, -0.79309423],
                    vec![-0.45778525, -0.92007301, 0.91021779],
                    vec![-0.57811735, 0.93723469, 0.22603182],
                    vec![-0.10960233, -0.49248161, -0.32769214],
                ])),
                // hidden to hidden weights
                Layer::new(Matrix::new(vec![
                    vec![-0.18527392, -0.77636724, 0.48528415, -0.32014238],
                    vec![-0.44772873, 0.44900513, -0.57772342, -0.08835226],
                    vec![0.51559826, 0.32360245, 0.88738762, 0.17788368],
                    vec![0.21927963, -0.36734758, 0.28202263, -0.46584744],
                ])),
                // hidden to output weights
                Layer::new(Matrix::new(vec![
                    vec![0.7376723, -0.87429221, 0.03202698, 0.19527884],
                    vec![0.01802741, -0.25608505, -0.37182482, 1.28308142],
                    vec![0.3852866, -0.37808645, -0.309396, -0.6621738],
                ])),
            ],
            transform_input_data_to_matrix(&input),
        )
        .unwrap();
        let expected_result = vec![
            Matrix::new(vec![vec![1.0], vec![0.5], vec![-1.5]]),
            Matrix::new(vec![
                vec![0.5744519553199438],
                vec![0.0925282494022273],
                vec![0.3897061104950745],
                vec![0.5338718107891272],
            ]),
            Matrix::new(vec![
                vec![0.460069301717884],
                vec![0.38037328757090877],
                vec![0.6828534410086781],
                vec![0.48829681532618874],
            ]),
            Matrix::new(vec![
                vec![0.5309719815497981],
                vec![0.5704062076991225],
                vec![0.37726694502286784],
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
