pub(crate) static MNIST_VERIFICATION_IMAGES: &'static [u8] =
    include_bytes!("../../mnist-dataset/t10k-images.idx3-ubyte");
pub(crate) static MNIST_TRAINING_IMAGES: &'static [u8] =
    include_bytes!("../../mnist-dataset/train-images.idx3-ubyte");
pub(crate) static MNIST_TRAINING_LABELS: &'static [u8] =
    include_bytes!("../../mnist-dataset/train-labels.idx1-ubyte");