use anyhow::Result;
use crate::{logic::feedforward::Feedforward, matrix::matrix::Matrix};
use super::layer::Layer;

#[derive(Debug)]
pub struct NeuralNetwork(Vec<Layer>);

impl NeuralNetwork {
    pub fn new(
        amount_of_input_neurons: u32,
        amount_of_hidden_neurons: u32,
        amount_of_output_neurons: u32,
    ) -> NeuralNetwork {
        NeuralNetwork(vec![
            Layer::new(Matrix::new_with_random_values(
                amount_of_hidden_neurons,
                amount_of_input_neurons,
            )),
            Layer::new(Matrix::new_with_random_values(
                amount_of_output_neurons,
                amount_of_hidden_neurons,
            )),
        ])
    }

    pub fn query(&self, input_data: Vec<f64>) -> Result<Vec<f64>> {
        Ok(Feedforward::run(&self.0, input_data)?.results)
    }
}
