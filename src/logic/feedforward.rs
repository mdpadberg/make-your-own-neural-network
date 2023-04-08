use crate::neuralnetwork::layer::Layer;
use anyhow::{bail, Context, Result};
use crate::matrix::matrix::Matrix;

#[derive(Debug, PartialEq)]
pub struct Feedforward {
    pub results: Vec<f64>,
}

impl Feedforward {
    pub fn run(layers: &Vec<Layer>, input_data: Vec<f64>) -> Result<Feedforward> {
        let amount_of_input_nodes: usize = layers
            .first()
            .context("Feedforward: no first layer")?
            .get_matrix_as_ref()
            .get_data_as_ref()
            .first()
            .context("Feedforward: no data in first layer")?
            .len();
        if amount_of_input_nodes == input_data.len() {
            bail!("Feedforward: This should not be possible, the input data should have the same size as the amount of input neurons")
        }
        if layers.len() < 2 {
            bail!("Feedforward: This should not be possible, there should always be an input, hidden, and output layer")
        }
        let end_result = run(layers, input_data, &mut None)?.0;
        let amount_of_output_nodes: usize = layers
            .last()
            .context("Feedforward: no last layer")?
            .get_matrix_as_ref()
            .get_data_as_ref()
            .first()
            .context("Feedforward: no data in last layer")?
            .len();
        if amount_of_output_nodes == end_result.results.len() {
            bail!("Feedforward: This should not be possible, the result should have the same size as the amount of output neurons")
        }
        Ok(end_result)
    }
}

fn run<'a>(
    layers: &Vec<Layer>,
    input_data: Vec<f64>,
    in_between_results: &'a mut Option<Vec<Matrix>>,
) -> Result<(Feedforward, &'a Option<Vec<Matrix>>)> {
    let mut result: Matrix = Matrix::new(input_data.into_iter().map(|value| vec![value]).collect::<Vec<Vec<f64>>>());
    for layer in layers {
        let mut before_activation_function: Matrix = layer.get_matrix_as_ref().matrix_multiplication(&result);
        if let Some(some) = in_between_results {
            some.push(before_activation_function.clone());
        }
        before_activation_function.apply_activation_function();
        if let Some(some) = in_between_results {
            some.push(before_activation_function.clone());
        }
        result = before_activation_function;
    }
    Ok((Feedforward { results: result.get_data().into_iter().flatten().collect() }, in_between_results))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn query_test_example_from_the_book() {
        let input = vec![0.9, 0.1, 0.8];
        let mut in_between_results = Some(vec![]);
        let actual_result = run(&vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![
                vec![0.9, 0.3, 0.4],
                vec![0.2, 0.8, 0.2],
                vec![0.1, 0.5, 0.6],
            ])),
            // hidden to output weights
            Layer::new(Matrix::new(vec![
                vec![0.3, 0.7, 0.5],
                vec![0.6, 0.5, 0.2],
                vec![0.8, 0.1, 0.9],
            ])),
        ], input, &mut in_between_results).unwrap();
        let expected_result =
            (vec![0.7263033450139793, 0.7085980724248232, 0.778097059561142],
             vec![
                 Matrix::new(vec![vec![1.1600000000000001], vec![0.42000000000000004], vec![0.62]]),
                 Matrix::new(vec![vec![0.7613327148429104], vec![0.6034832498647263], vec![0.6502185485738271]]),
                 Matrix::new(vec![vec![0.975947363645095], vec![0.8885849635528748], vec![1.2546111905772457]]),
                 Matrix::new(vec![vec![0.7263033450139793], vec![0.7085980724248232], vec![0.778097059561142]]),
             ]);
        assert_eq!(&actual_result.0.results, &expected_result.0);
        assert_eq!(&in_between_results.unwrap(), &expected_result.1);
    }

    /// Example
    /// 3 input nodes, 4 hidden nodes, again 4 hidden nodes, 3 output nodes
    /// Weight 1.1 means node 1 to node 1 in next layer
    /// Weight 1.2 means node 1 to node 2 in next layer
    /// Weight 2.1 means node 2 to node 1 in next layer
    #[test]
    fn query_test_my_own_example_multiple_hidden_layers_and_with_different_sizes() {
        let input = vec![0.9, 0.1, 0.8];
        let mut in_between_results = Some(vec![]);
        let actual_result = run(&vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1],
                vec![1.2, 2.2, 3.2],
                vec![1.3, 2.3, 3.3],
                vec![1.4, 2.4, 3.4],
            ])),
            // hidden to hidden weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
                vec![1.4, 2.4, 3.4, 4.4],
            ])),
            // hidden to output weights
            Layer::new(Matrix::new(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
            ])),
        ], input, &mut in_between_results).unwrap();
        let expected_result =
            (vec![0.9999695626612428, 0.9999795968549543, 0.9999863231488806],
             vec![
                 Matrix::new(vec![vec![3.6800000000000006], vec![3.8600000000000003], vec![4.04], vec![4.220000000000001]]),
                 Matrix::new(vec![vec![0.9753975715972605], vec![0.9793667027730938], vec![0.9827068434300946], vec![0.9855142760025332]]),
                 Matrix::new(vec![vec![10.216607150824164], vec![10.60890569020446], vec![11.001204229584758], vec![11.393502768965057]]),
                 Matrix::new(vec![vec![0.9999634432133694], vec![0.999975305509231], vec![0.9999833186780578], vec![0.9999887316656669]]),
                 Matrix::new(vec![vec![10.399810016835303], vec![10.799801096741938], vec![11.199792176648568]]),
                 Matrix::new(vec![vec![0.9999695626612428], vec![0.9999795968549543], vec![0.9999863231488806]]),
             ]);
        assert_eq!(&actual_result.0.results, &expected_result.0);
        assert_eq!(&in_between_results.unwrap(), &expected_result.1);
    }
}
