# Make your own neural network

## Online demo
[https://mdpadberg.github.io/make-your-own-neural-network/](https://mdpadberg.github.io/make-your-own-neural-network/)

## Project Info
This is my implement of a basic neural network with 0 dependencies. This implementation is inspired by the book "make your own neural network" of Tariq Rashid.

The core part of this project, is like the name implies, the center of this neural network project. This part does lots of matrix things, like: adding two matrices, matrix multiplications, transposing a matrix, etc. ***In this implementation I used 0 dependencies or libraries for any of the main functionalities.*** The core uses 4 dependencies: 2 for generating random numbers (rand & getrandom), 1 for making rust error handling a little easier (anyhow), and 1 for serializing and de-serializing the neural network to and from json. So I'm proud to say that this is my implementation, without the help of any libs :-)

The wasm part of this project is used to communicate between the core and the JavaScript frontend.

The www part of this project is the html, css, javascript to render everything in github pages. It communicates with wasm to query and/or train a neural network

## Performance
This isn't the most performant neural network implementation. Firstly, its only using the CPU while most professional neural network implementations are using the GPU. Secondly, its single threaded by design. While rust provides fearless concurrency, i didn’t feel that this was necessary in this project, because my goal was to learn how to create a simple neural network, not to make better tensorflow.

## Mnist dataset
The MNIST dataset is a database of handwritten digits. It has a training set of 60,000 examples and a test set of 10,000 examples. It is a subset of a larger set available from NIST. The digits have been size-normalized and centered in a fixed-size image. It is a good database for people who want to try learning techniques and pattern recognition methods on real-world data while spending minimal efforts on preprocessing and formatting. 

This data is from:
- Yann LeCun, Courant Institute, NYU
- Corinna Cortes, Google Labs, New York
- Christopher J.C. Burges, Microsoft Research, Redmond

For more information about the data and the license visit: [http://yann.lecun.com/exdb/mnist/](http://yann.lecun.com/exdb/mnist/)

## Run locally  

### Prerequisites
- rust
- rustup
- cargo
- wasm-bindgen

### How to run this project locally
1. run `./build.sh`
2. run `./run.sh`
3. visit `http://localhost:9000/`
