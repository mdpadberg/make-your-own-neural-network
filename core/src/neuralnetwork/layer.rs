use serde::{Serialize, Deserialize};

use crate::matrix::matrix::Matrix;

#[derive(Debug, PartialEq, Serialize, Deserialize)]
pub struct Layer(
    pub Matrix
);