use crate::mnist_image::MnistImage;
use anyhow::Context;
use image::{DynamicImage, ImageBuffer, ImageOutputFormat, Luma};
use std::{convert::TryFrom, io::Cursor};

/// This struct is used to convert a MnistImage to a base64 encoded png, so 
/// it can be displayed in the frontend
#[derive(Debug)]
pub(crate) struct Base64Png(pub(crate) String);

impl TryFrom<MnistImage> for Base64Png {
    type Error = anyhow::Error;

    fn try_from(value: MnistImage) -> Result<Self, Self::Error> {
        let dynamic_image = DynamicImage::ImageLuma8(
            ImageBuffer::<Luma<u8>, Vec<u8>>::from_raw(28, 28, value.0).context(
                "Could not create ImageBuffer, the container is not big enough. See Image crate",
            )?,
        );
        let mut buffer = Cursor::new(vec![]);
        dynamic_image.write_to(&mut buffer, ImageOutputFormat::Png)?;
        let result = base64::encode(&buffer.into_inner());
        Ok(Base64Png(format!(
            r#"<img src="data:image/png;base64,{result}">"#
        )))
    }
}
