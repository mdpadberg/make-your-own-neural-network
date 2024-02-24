use anyhow::Context;

use crate::{
    files::MNIST_TRAINING_LABELS, mnist_image::MnistImage, neuralnetwork_image::NeuralNetworkImage,
};
use core::neuralnetwork::{
    neuralnetwork::NeuralNetwork,
    training::{TrainingData, TrainingEntry},
};
use std::convert::TryFrom;

pub(crate) fn neural_network_from_string(
    neuralnetwork_as_string: String,
) -> anyhow::Result<NeuralNetwork> {
    Ok(serde_json::from_str(&neuralnetwork_as_string)?)
}

pub(crate) fn neural_network_to_string(neural_network: &NeuralNetwork) -> anyhow::Result<String> {
    Ok(serde_json::to_string(neural_network)?)
}

pub(crate) fn create(amount_of_hidden_neurons: u32) -> NeuralNetwork {
    NeuralNetwork::new_with_random_values(784, amount_of_hidden_neurons, 10, 1)
}

pub(crate) fn train(
    neural_network: NeuralNetwork,
    amount_of_training_rounds: u32,
    learning_rate: f64,
) -> anyhow::Result<NeuralNetwork> {
    let test_images: Vec<NeuralNetworkImage> = MnistImage::get_all_test_images()
        .into_iter()
        .map(NeuralNetworkImage::try_from)
        .collect::<Result<Vec<NeuralNetworkImage>, _>>()
        .context("train: cannot convert to NeuralNetworkImage")?;
    //Skip first bytes, thats the meta data, check README.md inside the mnist-dataset folder
    let test_labels: Vec<Vec<f64>> = MNIST_TRAINING_LABELS[8..]
        .iter()
        .map(|value| {
            let mut labels = vec![0.01; 10];
            labels[*value as usize] = 0.99;
            labels
        })
        .collect::<Vec<Vec<f64>>>();
    let training_data = TrainingData(
        test_images
            .into_iter()
            .zip(test_labels.into_iter())
            .map(|(image, label)| TrainingEntry {
                input: image.0,
                expected_output: label,
            })
            .collect::<Vec<TrainingEntry>>(),
    );
    neural_network.train(&training_data, amount_of_training_rounds, learning_rate)
}
