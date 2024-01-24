use rand::Rng;

static IMAGES: &'static [u8] = include_bytes!("../../mnist-dataset/t10k-images.idx3-ubyte");

pub(crate) struct MnistImage(pub(crate) Vec<u8>);

impl MnistImage {
    pub(crate) fn get_random() -> Self {
        //Skip first bytes, thats the meta data, check README.md inside the mnist-dataset folder
        let all_images = &IMAGES[16..];
        let mut rng = rand::thread_rng();
        MnistImage(
            all_images
            .chunks(784)
            // take a random image between 0 and 10000 (the size of the mnist verification dataset)
            // if we skip 0 we take the first image for the dataset, if we skip 9999 we take the last image from the dataset
            .skip(rng.gen_range(0..=9999))
            .take(1)
            .flat_map(|image| image.iter().map(|value| *value).collect::<Vec<u8>>())
            .collect::<Vec<u8>>()
        )
    }
}