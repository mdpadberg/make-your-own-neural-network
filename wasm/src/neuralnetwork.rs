use core::neuralnetwork::neuralnetwork::NeuralNetwork;

use anyhow::Context;

pub fn from_string_to_neuralnetwork(input: Option<String>) -> anyhow::Result<NeuralNetwork> {
    let input = input.context("Rust from_string_to_neuralnetwork input string is empty")?;
    Ok(serde_json::from_str(&input)?)
}
