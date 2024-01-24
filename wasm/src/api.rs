use crate::{
    base64_png::Base64Png, mnist_image::MnistImage, neuralnetwork::from_file,
    neuralnetwork_image::NeuralNetworkImage,
};
use core::neuralnetwork::query::{QueryData, QueryEntry};
use std::convert::TryFrom;
use wasm_bindgen::{prelude::wasm_bindgen, JsValue};

#[wasm_bindgen]
pub fn get_random_image() -> Result<String, JsValue> {
    match Base64Png::try_from(MnistImage::get_random()) {
        Ok(ok) => Ok(ok.0),
        Err(err) => {
            return Err(JsValue::from(format!(
                "Rust error in get_3_random_images: {:?}",
                err
            )))
        }
    }
}

#[wasm_bindgen]
pub fn query_neuralnetwork(image: String) -> Result<Vec<String>, JsValue> {
    match query_nn(image) {
        Ok(ok) => Ok(ok),
        Err(err) => {
            return Err(JsValue::from(format!(
                "Rust error in feed_to_neuralnetwork: {:?}",
                err
            )))
        }
    }
}

fn query_nn(image: String) -> anyhow::Result<Vec<String>> {
    let nn = from_file()?;
    let nn_image = NeuralNetworkImage::try_from(Base64Png(image))?;
    let result = nn.query(&QueryData(&vec![QueryEntry { input: nn_image.0 }]))?;
    Ok(result
        .0
        .iter()
        .flat_map(|queryresult| queryresult.0.iter().map(|result| result.to_string()))
        .collect::<Vec<String>>())
}
