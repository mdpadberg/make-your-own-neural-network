import { train_neuralnetwork } from "./pkg/wasm.js";

document.getElementById('train-neural-network').addEventListener('click', create);

function create(event) {
    event.preventDefault();
    const amountOfHiddenNeurons = document.getElementById('hidden-neurons').value;
    const amountOfTrainingRounds = document.getElementById('training-rounds').value;
    const learningRate = document.getElementById('learning-rate').value;
    try {
        train_neuralnetwork(amountOfHiddenNeurons, amountOfTrainingRounds, learningRate);
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}