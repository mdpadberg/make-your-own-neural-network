# Mnist-dataset
This data is from:
- Yann LeCun, Courant Institute, NYU
- Corinna Cortes, Google Labs, New York
- Christopher J.C. Burges, Microsoft Research, Redmond

For more information about the data and the license visit: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Files
| Name                    | Content             |
|-------------------------|---------------------|
| Train-labels-idx1-ubyte | Training set labels |
| Train-images-idx3-ubyte | Training set images |
| T10k-labels-idx1-ubyte  | Test set labels     |
| T10k-images-idx3-ubyte  | Test set images     |

## Train-labels-idx1-ubyte
| offset | type           | value           | description              |
|--------|----------------|-----------------|--------------------------|
| 0000   | 32 bit integer | 0x00000801(2049)| magic number (MSB first) |
| 0004   | 32 bit integer | 60000           | number of items          |
| 0008   | unsigned byte  | ??              | label                    |
| 0009   | unsigned byte  | ??              | label                    |
| .......|                |                 |                          |
| xxxx   | unsigned byte  | ??              | label                    |

The labels values are 0 to 9. 

## Train-images-idx3-ubyte
| offset | type           | value           | description              |
|--------|----------------|-----------------|--------------------------|
| 0000   | 32 bit integer | 0x00000803(2051)| magic number             |
| 0004   | 32 bit integer | 60000           | number of images         |
| 0008   | 32 bit integer | 28              | number of rows           |
| 0012   | 32 bit integer | 28              | number of columns        |
| 0016   | unsigned byte  | ??              | pixel                    |
| 0017   | unsigned byte  | ??              | pixel                    |
| .......|                |                 |                          |
| xxxx   | unsigned byte  | ??              | pixel                    |

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 

## T10k-labels-idx1-ubyte
| offset | type           | value           | description              |
|--------|----------------|-----------------|--------------------------|
| 0000   | 32 bit integer | 0x00000801(2049)| magic number (MSB first) |
| 0004   | 32 bit integer | 60000           | number of items          |
| 0008   | unsigned byte  | ??              | label                    |
| 0009   | unsigned byte  | ??              | label                    |
| .......|                |                 |                          |
| xxxx   | unsigned byte  | ??              | label                    |

The labels values are 0 to 9. 

## T10k-images-idx3-ubyte
| offset | type           | value           | description              |
|--------|----------------|-----------------|--------------------------|
| 0000   | 32 bit integer | 0x00000803(2051)| magic number             |
| 0004   | 32 bit integer | 60000           | number of images         |
| 0008   | 32 bit integer | 28              | number of rows           |
| 0012   | 32 bit integer | 28              | number of columns        |
| 0016   | unsigned byte  | ??              | pixel                    |
| 0017   | unsigned byte  | ??              | pixel                    |
| .......|                |                 |                          |
| xxxx   | unsigned byte  | ??              | pixel                    |

Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black). 