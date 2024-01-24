use crate::base64_png::Base64Png;
use image::EncodableLayout;
use std::convert::TryFrom;

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
