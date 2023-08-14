#[derive(Debug, Clone)]
pub struct ErrorRateEntry {
    pub input: Vec<f64>,
    pub expected_output: Vec<f64>
}

#[derive(Debug)]
pub struct ErrorRateData<'a>(pub &'a Vec<ErrorRateEntry>);
