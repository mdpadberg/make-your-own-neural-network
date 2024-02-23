use crate::files::PRE_TRAINED_NEURAL_NETWORK;
use core::neuralnetwork::neuralnetwork::NeuralNetwork;

pub(crate) fn load_pre_trained_neural_network() -> anyhow::Result<NeuralNetwork> {
    Ok(serde_json::from_str(&String::from_utf8(
        PRE_TRAINED_NEURAL_NETWORK.to_vec(),
    )?)?)
}

pub(crate) fn create(amount_of_hidden_neurons: u32) -> NeuralNetwork {
    NeuralNetwork::new_with_random_values(784, amount_of_hidden_neurons, 10, 1)
}

pub(crate) fn train(
    neural_network: NeuralNetwork,
    amount_of_training_rounds: u32,
    learning_rate: f64,
) -> anyhow::Result<String> {
    // let training_data = TrainingData(vec![
    //     TrainingEntry {
    //         input: vec![0.0; 784],
    //         expected_output: vec![0.0; 10],
    //     },
    // ]);
    // neural_network.train(&training_data, amount_of_training_rounds, learning_rate);
    Ok(String::from("Training complete"))
}
