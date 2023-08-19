use core::neuralnetwork::query::{QueryData, QueryEntry};

use crate::{image::from_string_to_f64_array, neuralnetwork::from_string_to_neuralnetwork};
use wasm_bindgen::{JsValue, prelude::wasm_bindgen};

#[wasm_bindgen]
pub fn query(neuralnetwork: JsValue, data: JsValue) -> Result<String, JsValue> {
    let user_drawing: Result<Vec<f64>, anyhow::Error> = from_string_to_f64_array(data.as_string());
    let neuralnetwork = from_string_to_neuralnetwork(neuralnetwork.as_string());

    match (user_drawing, neuralnetwork) {
        (Ok(user_drawing), Ok(neuralnetwork)) => {
            match neuralnetwork.query(&QueryData(&vec![QueryEntry {
                input: user_drawing,
            }])) {
                Ok(ok) => {
                    return Ok(ok
                        .0
                        .iter()
                        .map(|value| format!("{:?}", value.0))
                        .collect::<String>())
                }
                Err(err) => return Err(JsValue::from(format!("rust error in query: {:?}", err))),
            };
        }
        (Err(err), _) => return Err(JsValue::from(format!("rust error in query: {:?}", err))),
        (_, Err(err)) => return Err(JsValue::from(format!("rust error in query: {:?}", err))),
    }
}
