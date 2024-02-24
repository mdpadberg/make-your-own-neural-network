import { get_random_image, query_neuralnetwork } from "./pkg/wasm.js";

document.getElementById('get3images').addEventListener('click', get3images);
document.getElementById('guess').addEventListener('click', guessing);

function get3images(event) {
    event.preventDefault();
    try {
        document.getElementById('error-message-text').innerText = '';
        document.getElementById('imageone').innerHTML = `<input type="radio" id="one" name="image" value="one" checked />${get_random_image()}`;
        document.getElementById('imagetwo').innerHTML = `<input type="radio" id="two" name="image" value="two" />${get_random_image()}`;
        document.getElementById('imagethree').innerHTML = `<input type="radio" id="three" name="image" value="three" />${get_random_image()}`;
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}

function guessing(event) {
    event.preventDefault();
    try {
        document.getElementById('error-message-text').innerText = '';
        const selector = document.querySelector('input[name="image"]:checked');
        if (selector) {
            const neuralnetwork = document.getElementById("uploadInput").files[0];
            const reader = new FileReader();
            reader.readAsText(neuralnetwork);
            reader.onload = (function (f) {
                return function (e) {
                    const neuralnetworkAsText = e.target.result;
                    const selectedMnistImage = selector.nextSibling.outerHTML;
                    const result = query_neuralnetwork(neuralnetworkAsText, selectedMnistImage);
                    const resulttable = document.getElementById("resulttable");
                    let tabledata = "<tr><td>Number</td><td>%</td></tr>";
                    for (var i = 0; i < result.length; i++) {
                        tabledata += `<tr><td>${i}</td><td>${result[i]}</td></tr>`;
                    };
                    resulttable.innerHTML = tabledata;
                };
            })(neuralnetwork);
        };
    } catch (e) {
        document.getElementById('error-message-text').innerText = e;
    }
}