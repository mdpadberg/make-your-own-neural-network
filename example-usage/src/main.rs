use anyhow::Result;
use std::{fs::File, io::Read};

use lib::neuralnetwork::{
    errorrate::{ErrorRateData, ErrorRateEntry},
    neuralnetwork::NeuralNetwork,
    query::{QueryData, QueryEntry},
    training::{TrainingData, TrainingEntry},
};

const TRAINING_LABELS: &str = "train-labels.idx1-ubyte";
const TRAINING_IMAGES: &str = "train-images.idx3-ubyte";
const TEST_LABELS: &str = "t10k-labels.idx1-ubyte";
const TEST_IMAGES: &str = "t10k-images.idx3-ubyte";

#[derive(Debug)]
struct Image {
    label: Vec<f64>,
    datapoints: Vec<f64>,
}

fn main() -> Result<()> {
    // Querying and training becomes slower the more hidden neurons and hidden layers the network has
    let mut nn = NeuralNetwork::new_with_random_values(784, 25, 10, 3);
    let testing_data = get_mnist_training_labels_and_images(TEST_LABELS, TEST_IMAGES)
        .into_iter()
        .map(|image| ErrorRateEntry {
            input: image.datapoints,
            expected_output: image.label,
        })
        .collect::<Vec<ErrorRateEntry>>();
    let training_data = get_mnist_training_labels_and_images(TRAINING_LABELS, TRAINING_IMAGES)
        .into_iter()
        .map(|image| TrainingEntry {
            input: image.datapoints,
            expected_output: image.label,
        })
        .collect::<Vec<TrainingEntry>>();
    let error_rate_before_training = nn.error_rate_of_network(&ErrorRateData(&testing_data))?;
    nn = nn.train(&TrainingData(training_data), 1, 0.3)?;
    let error_rate_after_training = nn.error_rate_of_network(&ErrorRateData(&testing_data))?;
    println!("error_rate_before_training: {}", error_rate_before_training);
    println!("error_rate_after_training: {}", error_rate_after_training);
    Ok(())
}

// Returns a vector of Images. The image has a label which is the number which is in datapoints.
// An example of a number 7 is:
// Image {
//   label: [0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
//   datapoints: [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 84, 185,
// 159, 151, 60, 36, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 222, 254, 254, 254,
// 254, 241, 198, 198, 198, 198, 198, 198, 198, 198, 170, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 67, 114,
// 72, 114, 163, 227, 254, 225, 254, 254, 254, 250, 229, 254, 254, 140, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 17, 66, 14, 67, 67, 67, 59, 21, 236, 254, 106, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 83, 253, 209, 18, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 22, 233, 255, 83, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 129,
// 254, 238, 44, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 59, 249, 254, 62, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 133, 254, 187, 5, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 9, 205, 248, 58, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 126, 254, 182, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 75, 251, 240, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 19, 221, 254,
// 166, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 203, 254, 219, 35, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 254, 77, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 31, 224, 254, 115, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 133, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 61,
// 242, 254, 254, 52, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 254, 219, 40,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 121, 254, 207, 18, 0, 0, 0, 0, 0, 0, 0, 0,
// 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
// }
fn get_mnist_training_labels_and_images(
    label_file_name: &str,
    image_file_name: &str,
) -> Vec<Image> {
    let mut labels = File::open(format!(
        "{}/mnist-dataset/{}",
        env!("CARGO_MANIFEST_DIR").replace("/example-usage", ""),
        label_file_name
    ))
    .unwrap();
    let mut images = File::open(format!(
        "{}/mnist-dataset/{}",
        env!("CARGO_MANIFEST_DIR").replace("/example-usage", ""),
        image_file_name
    ))
    .unwrap();
    let mut buffer_1 = Vec::new();
    let mut buffer_2 = Vec::new();
    labels.read_to_end(&mut buffer_1).unwrap();
    images.read_to_end(&mut buffer_2).unwrap();
    //first range is meta data, check README.md inside the mnist-dataset folder
    buffer_1.drain(0..8);
    //first range is meta data, check README.md inside the mnist-dataset folder
    buffer_2.drain(0..16);
    buffer_2
        .chunks(784)
        .zip(buffer_1)
        .map(|(image, label)| {
            let mut labels = vec![0.0; 10];
            labels[label as usize] += 1.0;
            Image {
                label: labels,
                datapoints: image.iter().map(|f| *f as f64).collect(),
            }
        })
        .collect::<Vec<Image>>()
}
