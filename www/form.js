import { query } from "./pkg/wasm.js";

document.getElementById('submitbutton').addEventListener('click', submit)

function submit(event) {
    event.preventDefault();
    const drawing = canvas.toDataURL();
    const neuralnetwork = document.getElementById("uploadInput").files[0];
    const reader = new FileReader();
    reader.readAsText(neuralnetwork);
    reader.onload = (function (f) {
        return function (e) {
            const neuralnetworkAsText = e.target.result;
            console.log(query(neuralnetworkAsText, drawing));
        };
    })(neuralnetwork);
}