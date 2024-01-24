use core::neuralnetwork::neuralnetwork::NeuralNetwork;

static FILE: &'static [u8] = include_bytes!("./example-nn.txt");

pub(crate) fn from_file() -> anyhow::Result<NeuralNetwork> {
    Ok(serde_json::from_str(&String::from_utf8(FILE.to_vec())?)?)
}