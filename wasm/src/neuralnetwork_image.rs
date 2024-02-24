use crate::{base64_png::Base64Png, mnist_image::MnistImage};
use image::EncodableLayout;
use std::convert::TryFrom;

/// This struct is used to convert a MnistImage to a NeuralNetworkImage, so it 
/// can be used in the neural network
#[derive(Debug, Clone)]
pub(crate) struct NeuralNetworkImage(pub(crate) Vec<f64>);

impl TryFrom<Base64Png> for NeuralNetworkImage {
    type Error = anyhow::Error;

    fn try_from(value: Base64Png) -> Result<Self, Self::Error> {
        let base64_encoded_png = &value.0[32..].replace(r#"">"#, "");
        let decoded_image = base64::decode(base64_encoded_png)?;
        let image = image::load_from_memory_with_format(&decoded_image, image::ImageFormat::Png)?;
        Ok(NeuralNetworkImage(
            image
                .to_luma8() // same as mnist dataset
                .as_bytes()
                .iter()
                //normalize the values because of sigmoid (between 0 and 1)
                .map(|value| (((*value as f64) / 255.0) * 0.99) + 0.01)
                .collect::<Vec<f64>>(),
        ))
    }
}

impl TryFrom<MnistImage> for NeuralNetworkImage {
    type Error = anyhow::Error;

    fn try_from(value: MnistImage) -> Result<Self, Self::Error> {
        Ok(NeuralNetworkImage(
            value
                .0
                .iter()
                //normalize the values because of sigmoid (between 0 and 1)
                .map(|value| (((*value as f64) / 255.0) * 0.99) + 0.01)
                .collect::<Vec<f64>>(),
        ))
    }
}
