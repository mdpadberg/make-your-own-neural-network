import { train_neuralnetwork } from "./pkg/wasm.js";

document.getElementById('train-neural-network').addEventListener('click', train);

function train(event) {
    event.preventDefault();
    const amountOfHiddenNeurons = document.getElementById('hidden-neurons').value;
    const amountOfTrainingRounds = document.getElementById('training-rounds').value;
    const learningRate = document.getElementById('learning-rate').value;
    try {
        document.getElementById('error-message-text').innerText = '';
        const trainedNeuralNetwork = train_neuralnetwork(amountOfHiddenNeurons, amountOfTrainingRounds, learningRate);
        download('your-neural-network.txt', trainedNeuralNetwork);
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}

function download(filename, text) {
    const element = document.createElement('a');
    element.setAttribute('href', 'data:text/plain;charset=utf-8,' + encodeURIComponent(text));
    element.setAttribute('download', filename);
    element.style.display = 'none';
    document.body.appendChild(element);
    element.click();
    document.body.removeChild(element);
}