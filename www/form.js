import { query } from "./pkg/wasm.js";

document.getElementById('submitbutton').addEventListener('click', submit);

function submit(event) {
    event.preventDefault();
    const drawing = canvas.toDataURL();
    const neuralnetwork = document.getElementById("uploadInput").files[0];
    const reader = new FileReader();
    reader.readAsText(neuralnetwork);
    reader.onload = (function (f) {
        return function (e) {
            const neuralnetworkAsText = e.target.result;
            const result = query(neuralnetworkAsText, drawing);
            const resulttable = document.getElementById("resulttable");
            let tabledata = "<tr><td>Number</td><td>%</td></tr>";
            for(var i = 0; i < result.length; i++) {
                tabledata += `<tr><td>${i}</td><td>${result[i]}</td></tr>`;
            };
            resulttable.innerHTML = tabledata;
        };
    })(neuralnetwork);
}