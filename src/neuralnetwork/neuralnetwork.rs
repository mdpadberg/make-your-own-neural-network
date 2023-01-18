#[derive(Debug)]
pub struct neuralnetwork {
    layers: Vec<Layer>
}

#[derive(Debug)]
pub struct Layer(Vec<f64>);

#[derive(Debug)]
pub struct Weight(f64);

impl neuralnetwork {
    fn new(
        amount_of_input_neurons: u32,
        amount_of_hidden_neurons: u32,
        amount_of_output_neurons: u32,
    ) -> neuralnetwork {
        neuralnetwork {
            layers: vec![]
        }
    }
}
