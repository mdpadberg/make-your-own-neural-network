import { get_random_image, query_neuralnetwork } from "./pkg/wasm.js";

document.getElementById('get3images').addEventListener('click', get3images);
document.getElementById('guess').addEventListener('click', guessing);

function get3images(event) {
    event.preventDefault();
    try {
        document.getElementById('error-message-text').innerText = '';
        document.getElementById('imageone').innerHTML = imageElement(true, 'one');
        document.getElementById('imagetwo').innerHTML = imageElement(false, 'two');
        document.getElementById('imagethree').innerHTML = imageElement(false, 'three');
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}

function imageElement(checked, id) {
    return `<input type="radio" id="${id}" name="image" value="${id}" ${checked ? 'checked' : ''}/>${get_random_image()}`
}

function guessing(event) {
    event.preventDefault();
    try {
        document.getElementById('error-message-text').innerText = '';
        const selector = document.querySelector('input[name="image"]:checked');
        if (selector) {
            const neuralnetwork = document.getElementById('uploadInput').files[0];
            if(!neuralnetwork) {
                throw new Error("You didn't upload a neural network");
            }
            const reader = new FileReader();
            reader.readAsText(neuralnetwork);
            reader.onload = (function (f) {
                return function (e) {
                    const neuralnetworkAsText = e.target.result;
                    const selectedMnistImage = selector.nextSibling.outerHTML;
                    const result = query_neuralnetwork(neuralnetworkAsText, selectedMnistImage);
                    const resulttable = document.getElementById('resulttable');
                    let tabledata = '<tr><td>Number</td><td>%</td></tr>';
                    for (var i = 0; i < result.length; i++) {
                        tabledata += `<tr><td>${i}</td><td>${result[i]*100}</td></tr>`;
                    };
                    resulttable.innerHTML = tabledata;
                };
            })(neuralnetwork);
        };
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}