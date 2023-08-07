use crate::logic::feedforward::Feedforward;
use crate::matrix::matrix::Matrix;
use crate::neuralnetwork::layer::Layer;
use anyhow::{Context, Result};

#[derive(Debug)]
pub(crate) struct Backpropagation {
    new_layers: Vec<Layer>,
}

#[derive(Debug)]
struct ErrorRatePerLayer(Vec<Matrix>);

impl Backpropagation {
    pub(crate) fn run(target: &Vec<f64>, layers: &Vec<Layer>, feedforward: &Feedforward) {}
}

/// calculate_error_rate_per_layer:
/// This method returns the error rate per layer, by calculating the error rate from the
/// output layer (target - actual). Then going through each layer from the back to the front,
/// while skipping the input layer. This way you can see how much each layer contributed to
/// the error rate of the neural network
fn calculate_error_rate_per_layer(
    target: &Matrix,
    actual: &Matrix,
    layers: &Vec<Layer>,
) -> Result<ErrorRatePerLayer> {
    let mut result: Vec<Matrix> = vec![error_rate_from_last_layer(&target, &actual)?];
    for layer in layers.iter().skip(1).rev() {
        let result_from_last_processed_layer =
            result.last().context("Backpropagation: No last layer")?;
        let current_matrix = layer.0.transpose();
        result.push((current_matrix * result_from_last_processed_layer)?);
    }
    result.reverse();
    Ok(ErrorRatePerLayer(result))
}

/// new_weights_based_on_error_rate_and_gradient_descent
/// This method will returns the updated weights for each layer. With the help of the error rate
/// and gradient descent. By changing the weights in incremental steps, the neural network will try
/// to find the minimum error rate.
///
/// The formula from the book:
/// ∆Wih = α * Eh * sigmoid (Oh) * (1- sigmoid (Oh)) * OiT
///
/// Meaning:
///  ∆Wih = delta of the weights between input and the hidden layer
///  i = input layer
///  h = hidden layer
///  α = learning rate
///  Eh = error rate hidden layer
///  Oh = output of hidden layer before the activation function
///  sigmoid () = activation function
///  sigmoid (Oh) * (1- sigmoid (Oh)) = derivative of activation function
///  OiT = output of input layer after activation function transposed
fn new_weights_based_on_error_rate_and_gradient_descent(
    learning_rate: &f64,
    feedforward: &Feedforward,
    layers: &Vec<Layer>,
    error_rate_per_layer: &ErrorRatePerLayer,
) -> Result<Vec<Layer>> {
    let mut new_layers = vec![];
    for (i, layer) in layers.iter().enumerate() {
        let weight_adjustments = learning_rate
            * ((&error_rate_per_layer.0[i]
                * &feedforward.results[i + 1].derivative_of_sigmoid())?
                * &feedforward.results[i].transpose())?;
        let old_layer_matrix = &layer.0;
        let new_matrix = (old_layer_matrix + &weight_adjustments)?;
        new_layers.push(Layer(new_matrix));
    }
    Ok(new_layers)
}

fn error_rate_from_last_layer(target: &Matrix, actual: &Matrix) -> Result<Matrix> {
    Ok((target - actual)?)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuralnetwork::neuralnetwork::NeuralNetwork;

    #[test]
    fn calculate_error_rate_example_from_the_book() {
        let layers = vec![
            // input to hidden weights
            Layer(Matrix(vec![vec![0.42, 0.28], vec![0.16, 1.1375]])),
            Layer(Matrix(vec![
                vec![0.4333333333333333, 0.7333333333333334],
                vec![0.1, 0.4],
            ])),
        ];
        let actual = calculate_error_rate_per_layer(
            &Matrix(vec![vec![3.0], vec![3.0]]),
            &Matrix(vec![vec![1.5], vec![2.5]]),
            &layers,
        )
        .unwrap()
        .0;
        let expected = vec![
            Matrix(vec![vec![0.7], vec![1.3]]),
            Matrix(vec![vec![1.5], vec![0.5]]),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculate_error_rate_example_with_one_hidden_layers_and_with_different_sizes() {
        let layers = vec![
            // input to hidden weights
            Layer(Matrix(vec![
                vec![-0.62091563, -1.25154723, -0.14334102],
                vec![0.07702094, -0.04681276, 0.62532329],
                vec![0.64765397, -0.13824144, 0.04829786],
                vec![0.58148507, -0.63079071, -0.38690071],
            ])),
            // hidden to output weights
            Layer(Matrix(vec![
                vec![-0.12369625, 0.16732224, -0.02174357, -0.06230827],
                vec![0.32413962, 0.2632736, -0.01882388, 0.12343084],
                vec![0.23642887, -0.44747458, 0.14426986, -0.11192151],
            ])),
        ];
        let actual = calculate_error_rate_per_layer(
            &Matrix(vec![vec![1.0], vec![0.0], vec![0.0]]),
            &Matrix(vec![vec![0.50024943], vec![0.58025132], vec![0.46906931]]),
            &layers,
        )
        .unwrap()
        .0;
        let expected = vec![
            Matrix(vec![
                vec![-0.3608012407286406],
                vec![0.14075112337566462],
                vec![-0.0676163439678099],
                vec![-0.05026055581706461],
            ]),
            Matrix(vec![vec![0.49975057], vec![-0.58025132], vec![-0.46906931]]),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn calculate_error_rate_example_with_multiple_hidden_layers_and_with_different_sizes() {
        let layers = &vec![
            // input to hidden weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1],
                vec![1.2, 2.2, 3.2],
                vec![1.3, 2.3, 3.3],
                vec![1.4, 2.4, 3.4],
            ])),
            // hidden to hidden weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
                vec![1.4, 2.4, 3.4, 4.4],
            ])),
            // hidden to output weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
            ])),
        ];
        let actual = calculate_error_rate_per_layer(
            &Matrix(vec![vec![2.0], vec![3.0], vec![6.0]]),
            &Matrix(vec![
                vec![0.9999695369356685],
                vec![0.9999795788316486],
                vec![0.9999863105453931],
            ]),
            &layers,
        )
        .unwrap()
        .0;
        let expected = vec![
            Matrix(vec![
                vec![114.0008956448172],
                vec![202.00158633119605],
                vec![290.0022770175749],
                vec![378.0029677039537],
            ]),
            Matrix(vec![
                vec![10.000075811063777],
                vec![18.000140384751067],
                vec![26.000204958438356],
                vec![34.00026953212564],
            ]),
            Matrix(vec![
                vec![1.0000304630643315],
                vec![2.0000204211683514],
                vec![5.000013689454607],
            ]),
        ];
        assert_eq!(actual, expected);
    }

    #[test]
    fn new_weights_based_on_error_rate_and_gradient_descent_with_multiple_hidden_layers_and_with_different_sizes(
    ) {
        let layers = vec![
            // input to hidden weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1],
                vec![1.2, 2.2, 3.2],
                vec![1.3, 2.3, 3.3],
                vec![1.4, 2.4, 3.4],
            ])),
            // hidden to hidden weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
                vec![1.4, 2.4, 3.4, 4.4],
            ])),
            // hidden to output weights
            Layer(Matrix(vec![
                vec![1.1, 2.1, 3.1, 4.1],
                vec![1.2, 2.2, 3.2, 4.2],
                vec![1.3, 2.3, 3.3, 4.3],
            ])),
        ];
        let error_rate_per_layer = calculate_error_rate_per_layer(
            &Matrix(vec![vec![2.0], vec![3.0], vec![6.0]]),
            &Matrix(vec![
                vec![0.9999695369356685],
                vec![0.9999795788316486],
                vec![0.9999863105453931],
            ]),
            &layers,
        )
        .unwrap();
        let feedforward = Feedforward::run(
            NeuralNetwork {
                layers: vec![
                    // input to hidden weights
                    Layer(Matrix(vec![
                        vec![1.1, 2.1, 3.1],
                        vec![1.2, 2.2, 3.2],
                        vec![1.3, 2.3, 3.3],
                        vec![1.4, 2.4, 3.4],
                    ])),
                    // hidden to hidden weights
                    Layer(Matrix(vec![
                        vec![1.1, 2.1, 3.1, 4.1],
                        vec![1.2, 2.2, 3.2, 4.2],
                        vec![1.3, 2.3, 3.3, 4.3],
                        vec![1.4, 2.4, 3.4, 4.4],
                    ])),
                    // hidden to output weights
                    Layer(Matrix(vec![
                        vec![1.1, 2.1, 3.1, 4.1],
                        vec![1.2, 2.2, 3.2, 4.2],
                        vec![1.3, 2.3, 3.3, 4.3],
                    ])),
                ],
                amount_of_input_neurons: 3,
                amount_of_hidden_neurons: 4,
                amount_of_output_neurons: 3,
                amount_of_hidden_layers: 2,
            },
            &vec![0.1, 0.2, 0.3],
        )
        .unwrap();
        let expected = vec![
            // input to hidden weights
            Layer(Matrix(vec![
                vec![1.6230841920849253, 3.1461683841698505, 4.669252576254776],
                vec![2.0923728075635353, 3.9847456151270704, 5.8771184226906055],
                vec![2.5321418269215714, 4.764283653843143, 6.9964254807647155],
                vec![2.9430477971611615, 5.4860955943223235, 8.029143391483483],
            ])),
            // hidden to hidden weights
            Layer(Matrix(vec![
                vec![
                    1.1004390858706103,
                    2.100443958529723,
                    3.1004486473545705,
                    4.1004531545908,
                ],
                vec![
                    1.2005683299771444,
                    2.2005746368943733,
                    3.2005807058660634,
                    4.200586539798865,
                ],
                vec![
                    1.3005902929217656,
                    2.3005968435679534,
                    3.300603147073267,
                    4.300609206456684,
                ],
                vec![
                    1.400555047820614,
                    2.4005612073420246,
                    3.4005671344788033,
                    4.40057283206967,
                ],
            ])),
            // hidden to output weights
            Layer(Matrix(vec![
                vec![
                    1.1000091372707792,
                    2.100009137733936,
                    3.1000091380669783,
                    4.100009138306454,
                ],
                vec![
                    1.2000122503657362,
                    2.200012250986692,
                    3.2000122514332032,
                    4.200012251754269,
                ],
                vec![
                    1.3000205302530374,
                    2.3000205312936894,
                    3.300020532041993,
                    4.3000205325800644,
                ],
            ])),
        ];
        let actual = new_weights_based_on_error_rate_and_gradient_descent(
            &0.3,
            &feedforward,
            &layers,
            &error_rate_per_layer,
        )
        .unwrap();
        assert_eq!(actual, expected);
    }
}
