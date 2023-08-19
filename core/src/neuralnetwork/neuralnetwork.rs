use super::{
    errorrate::ErrorRateData,
    layer::Layer,
    query::{QueryData, QueryResult, QueryResults},
    training::TrainingData,
};
use crate::{
    logic::{backpropagation::Backpropagation, feedforward::Feedforward},
    matrix::matrix::Matrix,
};
use anyhow::{ensure, Context, Result};
use serde::{Serialize, Deserialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct NeuralNetwork {
    pub layers: Vec<Layer>,
    pub amount_of_input_neurons: u32,
    pub amount_of_hidden_neurons: u32,
    pub amount_of_output_neurons: u32,
    pub amount_of_hidden_layers: u32,
}

impl NeuralNetwork {
    pub fn new_with_random_values(
        amount_of_input_neurons: u32,
        amount_of_hidden_neurons: u32,
        amount_of_output_neurons: u32,
        amount_of_hidden_layers: u32,
    ) -> NeuralNetwork {
        let mut layers = Vec::with_capacity(amount_of_hidden_layers as usize + 2);
        layers.push(create_input_layer(
            amount_of_hidden_neurons,
            amount_of_input_neurons,
        ));
        for layer in create_hidden_layers(amount_of_hidden_layers, amount_of_hidden_neurons) {
            layers.push(layer);
        }
        layers.push(create_output_layer(
            amount_of_output_neurons,
            amount_of_hidden_neurons,
        ));
        NeuralNetwork {
            layers,
            amount_of_input_neurons,
            amount_of_hidden_neurons,
            amount_of_output_neurons,
            amount_of_hidden_layers,
        }
    }

    pub fn query(&self, input_data: &QueryData) -> Result<QueryResults> {
        let mut queryresults = Vec::new();
        for entry in input_data.0.iter() {
            queryresults.push(QueryResult(
                Feedforward::run(self, &entry.input)?
                    .results
                    .last()
                    .context("Query: result has no last layer")?
                    .0
                    .iter()
                    .flat_map(|a| a.to_owned())
                    .collect::<Vec<f64>>(),
            ));
        }
        Ok(QueryResults(queryresults))
    }

    pub fn error_rate_of_network(&self, input_data: &ErrorRateData) -> Result<f64> {
        let mut error_rate = 0.0;
        for entry in input_data.0.iter() {
            let actual_result = Feedforward::run(self, &entry.input)?
                .results
                .last()
                .context("Query: result has no last layer")?
                .0
                .iter()
                .flat_map(|a| a.to_owned())
                .collect::<Vec<f64>>();
            error_rate += entry
                .expected_output
                .iter()
                .zip(actual_result)
                .map(|(expected, actual)| expected - actual)
                .sum::<f64>();
        }
        Ok(error_rate)
    }

    pub fn train(
        self,
        training_data: &TrainingData,
        rounds: u32,
        learning_rate: f64,
    ) -> Result<NeuralNetwork> {
        let mut nn = self;
        for _ in 0..rounds {
            for entry in training_data.0.iter() {
                ensure!(
                    entry.input.len() == nn.amount_of_input_neurons as usize,
                    "Neuralnetwork: TrainingEntry input should be of same size as amount_of_input_neurons"
                );
                ensure!(
                    entry.expected_output.len() == nn.amount_of_output_neurons as usize,
                    "Neuralnetwork: TrainingEntry expected_output should be of same size as amount_of_output_neurons"
                );
                let feedforward = Feedforward::run(&nn, &entry.input)?;
                let backpropagation = Backpropagation::run(
                    &learning_rate,
                    &nn.layers,
                    &entry.expected_output,
                    &feedforward,
                )?;
                nn.layers = backpropagation.new_layers;
            }
        }
        Ok(nn)
    }
}

fn create_input_layer(amount_of_hidden_neurons: u32, amount_of_input_neurons: u32) -> Layer {
    Layer(Matrix::new_with_random_values(
        amount_of_hidden_neurons,
        amount_of_input_neurons,
    ))
}

fn create_hidden_layers(amount_of_hidden_layers: u32, amount_of_hidden_neurons: u32) -> Vec<Layer> {
    (0..amount_of_hidden_layers)
        .into_iter()
        .map(|_| {
            Layer(Matrix::new_with_random_values(
                amount_of_hidden_neurons,
                amount_of_hidden_neurons,
            ))
        })
        .collect::<Vec<Layer>>()
}

fn create_output_layer(amount_of_output_neurons: u32, amount_of_hidden_neurons: u32) -> Layer {
    Layer(Matrix::new_with_random_values(
        amount_of_output_neurons,
        amount_of_hidden_neurons,
    ))
}

#[cfg(test)]
mod test {
    use crate::{
        matrix::matrix::Matrix,
        neuralnetwork::{
            errorrate::{ErrorRateData, ErrorRateEntry},
            layer::Layer,
            neuralnetwork::NeuralNetwork,
            query::{QueryData, QueryEntry},
            training::{TrainingData, TrainingEntry},
        },
    };

    #[test]
    fn testing_new() {
        let nn = NeuralNetwork::new_with_random_values(3, 4, 3, 2);
        assert_eq!(nn.amount_of_input_neurons, 3);
        assert_eq!(nn.amount_of_hidden_neurons, 4);
        assert_eq!(nn.amount_of_output_neurons, 3);
        assert_eq!(nn.amount_of_hidden_layers, 2);
        assert_eq!(nn.layers.len(), 4);
        let input_layer = &nn.layers[0].0 .0;
        assert_eq!(input_layer.len(), 4);
        assert_eq!(input_layer[0].len(), 3);
        let hidden_layer_1 = &nn.layers[1].0 .0;
        assert_eq!(hidden_layer_1.len(), 4);
        assert_eq!(hidden_layer_1[0].len(), 4);
        let hidden_layer_2 = &nn.layers[2].0 .0;
        assert_eq!(hidden_layer_2.len(), 4);
        assert_eq!(hidden_layer_2[0].len(), 4);
        let output_layer = &nn.layers[3].0 .0;
        assert_eq!(output_layer.len(), 3);
        assert_eq!(output_layer[0].len(), 4);
    }

    #[test]
    fn testing_query() {
        let nn = NeuralNetwork {
            layers: vec![
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
            amount_of_input_neurons: 3,
            amount_of_hidden_neurons: 3,
            amount_of_output_neurons: 3,
            amount_of_hidden_layers: 1,
        };
        let actual_result = nn
            .query(&QueryData(&vec![QueryEntry {
                input: vec![0.9, 0.1, 0.8],
            }]))
            .unwrap();
        assert_eq!(
            actual_result.0.get(0).unwrap().0,
            vec![0.7263033450139793, 0.7085980724248232, 0.778097059561142]
        );
    }

    #[test]
    fn testing_error_rate() {
        let nn = NeuralNetwork {
            layers: vec![
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
            amount_of_input_neurons: 3,
            amount_of_hidden_neurons: 3,
            amount_of_output_neurons: 3,
            amount_of_hidden_layers: 1,
        };
        let actual_result = nn
            .error_rate_of_network(&ErrorRateData(&vec![ErrorRateEntry {
                input: vec![0.9, 0.1, 0.8],
                expected_output: vec![0.7263033450139793, 0.7085980724248232, 0.778097059561142],
            }]))
            .unwrap();
        assert_eq!(actual_result, 0.0);
    }

    #[test]
    fn testing_train() {
        let old_nn = NeuralNetwork {
            layers: vec![
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
            amount_of_input_neurons: 3,
            amount_of_hidden_neurons: 3,
            amount_of_output_neurons: 3,
            amount_of_hidden_layers: 1,
        };
        let input = vec![0.1, 0.2, 0.3];
        let new_nn = old_nn
            .train(
                &TrainingData(vec![TrainingEntry {
                    input: input.clone(),
                    expected_output: vec![0.5, 0.5, 0.5],
                }]),
                1,
                0.3,
            )
            .unwrap();
        let actual_result = new_nn
            .query(&QueryData(&vec![QueryEntry {
                input: input.clone(),
            }]))
            .unwrap();
        assert_eq!(
            actual_result.0.get(0).unwrap().0,
            vec![0.6973393995613739, 0.673130310893913, 0.7329538405694166]
        );
    }
}
