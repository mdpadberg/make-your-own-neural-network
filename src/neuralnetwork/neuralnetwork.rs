use anyhow::Result;

use crate::matrix::matrix::Matrix;

use super::layer::Layer;

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
        let mut neural_network = NeuralNetwork {
            layers: vec![],
            amount_of_input_neurons,
            amount_of_hidden_neurons,
            amount_of_output_neurons,
            amount_of_hidden_layers,
        };
        neural_network
            .layers
            .push(Layer::new(Matrix::new_with_random_values(
                amount_of_hidden_neurons,
                amount_of_input_neurons,
            )));
        for _ in 0..amount_of_hidden_layers {
            neural_network
                .layers
                .push(Layer::new(Matrix::new_with_random_values(
                    amount_of_hidden_neurons,
                    amount_of_hidden_neurons,
                )));
        }
        neural_network
            .layers
            .push(Layer::new(Matrix::new_with_random_values(
                amount_of_output_neurons,
                amount_of_hidden_neurons,
            )));
        neural_network
    }

    pub fn query(&self) -> Result<()> {
        Ok(())
    }

    pub fn train() -> Result<()> {
        Ok(())
    }

    pub fn layers(&self) -> &Vec<Layer> {
        &self.layers
    }

    pub fn amount_of_input_neurons(&self) -> &u32 {
        &self.amount_of_input_neurons
    }

    pub fn amount_of_hidden_neurons(&self) -> &u32 {
        &self.amount_of_hidden_neurons
    }

    pub fn amount_of_output_neurons(&self) -> &u32 {
        &self.amount_of_output_neurons
    }

    pub fn amount_of_hidden_layers(&self) -> &u32 {
        &self.amount_of_hidden_layers
    }
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
        let input_layer = nn.layers[0].get_matrix_as_ref().get_data_as_ref();
        assert_eq!(input_layer.len(), 4);
        assert_eq!(input_layer[0].len(), 3);
        let hidden_layer_1 = nn.layers[1].get_matrix_as_ref().get_data_as_ref();
        assert_eq!(hidden_layer_1.len(), 4);
        assert_eq!(hidden_layer_1[0].len(), 4);
        let hidden_layer_2 = nn.layers[2].get_matrix_as_ref().get_data_as_ref();
        assert_eq!(hidden_layer_2.len(), 4);
        assert_eq!(hidden_layer_2[0].len(), 4);
        let output_layer = nn.layers[3].get_matrix_as_ref().get_data_as_ref();
        assert_eq!(output_layer.len(), 3);
        assert_eq!(output_layer[0].len(), 4);
    }
}
