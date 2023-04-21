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
    fn feedforward_example_from_the_book() {
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

    #[test]
    fn feedforward_my_own_example_with_multiple_hidden_layers_and_with_different_sizes() {
        let input = vec![1.0, 0.5, -1.5];
        let mut in_between_results = Some(vec![]);
        let actual_result = run(&vec![
            // input to hidden weights
            Layer::new(Matrix::new(vec![
                vec![-1.03531205,  0.29141863, -0.79309423],
                vec![-0.45778525, -0.92007301,  0.91021779],
                vec![-0.57811735,  0.93723469,  0.22603182],
                vec![-0.10960233, -0.49248161, -0.32769214],
            ])),
            // hidden to hidden weights
            Layer::new(Matrix::new(vec![
                vec![-0.18527392, -0.77636724,  0.48528415, -0.32014238],
                vec![-0.44772873,  0.44900513, -0.57772342, -0.08835226],
                vec![ 0.51559826,  0.32360245,  0.88738762,  0.17788368],
                vec![ 0.21927963, -0.36734758,  0.28202263, -0.46584744],
            ])),
            // hidden to output weights
            Layer::new(Matrix::new(vec![
                vec![0.7376723 , -0.87429221,  0.03202698,  0.19527884],
                vec![0.01802741, -0.25608505, -0.37182482,  1.28308142],
                vec![0.3852866 , -0.37808645, -0.309396  , -0.6621738 ],
            ])),
        ], input, &mut in_between_results).unwrap();
        let expected_result =
            (vec![0.5309719815497981, 0.5704062076991225, 0.37726694502286784],
             vec![
                 Matrix::new(vec![vec![0.3000386100000001], vec![-2.28314844], vec![-0.44854773499999995], vec![0.13569507500000005]]),
                 Matrix::new(vec![vec![0.5744519553199438], vec![0.0925282494022273], vec![0.3897061104950745], vec![0.5338718107891272]]),
                 Matrix::new(vec![vec![-0.16006366076376224], vec![-0.4879641137335198], vec![0.7669162570604473], vec![-0.046821290419598594]]),
                 Matrix::new(vec![vec![0.460069301717884], vec![0.38037328757090877], vec![0.6828534410086781], vec![0.48829681532618874]]),
                 Matrix::new(vec![vec![0.12404674691299854], vec![0.2835086589649931], vec![-0.5011649299160179]]),
                 Matrix::new(vec![vec![0.5309719815497981], vec![0.5704062076991225], vec![0.37726694502286784]]),
             ]);
        assert_eq!(&actual_result.0.results, &expected_result.0);
        assert_eq!(&in_between_results.unwrap(), &expected_result.1);
    }
}
