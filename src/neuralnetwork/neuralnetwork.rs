use super::layer::Layer;
use crate::matrix::matrix::Matrix;
use anyhow::Result;

#[derive(Debug)]
pub struct NeuralNetwork {
    pub(crate) layers: Vec<Layer>,
    pub(crate) amount_of_input_neurons: u32,
    pub(crate) amount_of_hidden_neurons: u32,
    pub(crate) amount_of_output_neurons: u32,
    pub(crate) amount_of_hidden_layers: u32,
}

impl NeuralNetwork {
    pub fn new(
        amount_of_input_neurons: u32,
        amount_of_hidden_neurons: u32,
        amount_of_output_neurons: u32,
        amount_of_hidden_layers: u32,
    ) -> NeuralNetwork {
        let mut layers = Vec::with_capacity(amount_of_hidden_layers as usize + 2);
        layers.push(create_input_layer(amount_of_hidden_neurons, amount_of_input_neurons));
        for layer in create_hidden_layers(amount_of_hidden_layers, amount_of_hidden_neurons) {
            layers.push(layer);
        }
        layers.push(create_output_layer(amount_of_output_neurons, amount_of_hidden_neurons));
        NeuralNetwork {
            layers,
            amount_of_input_neurons,
            amount_of_hidden_neurons,
            amount_of_output_neurons,
            amount_of_hidden_layers,
        }
    }

    pub fn query(&self) -> Result<()> {
        Ok(())
    }

    pub fn train() -> Result<()> {
        Ok(())
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
    use crate::neuralnetwork::neuralnetwork::NeuralNetwork;

    #[test]
    fn creating_new_neural_network() {
        let nn = NeuralNetwork::new(3, 4, 3, 2);
        assert_eq!(nn.amount_of_input_neurons, 3);
        assert_eq!(nn.amount_of_hidden_neurons, 4);
        assert_eq!(nn.amount_of_output_neurons, 3);
        assert_eq!(nn.amount_of_hidden_layers, 2);
        assert_eq!(nn.layers.len(), 4);
        let input_layer = &nn.layers[0].0.0;
        assert_eq!(input_layer.len(), 4);
        assert_eq!(input_layer[0].len(), 3);
        let hidden_layer_1 = &nn.layers[1].0.0;
        assert_eq!(hidden_layer_1.len(), 4);
        assert_eq!(hidden_layer_1[0].len(), 4);
        let hidden_layer_2 = &nn.layers[2].0.0;
        assert_eq!(hidden_layer_2.len(), 4);
        assert_eq!(hidden_layer_2[0].len(), 4);
        let output_layer = &nn.layers[3].0.0;
        assert_eq!(output_layer.len(), 3);
        assert_eq!(output_layer[0].len(), 4);
    }
}
