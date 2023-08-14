#[derive(Debug)]
pub struct TrainingEntry {
    pub input: Vec<f64>,
    pub expected_output: Vec<f64>
}

#[derive(Debug)]
pub struct TrainingData(pub Vec<TrainingEntry>);