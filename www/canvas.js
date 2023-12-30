const canvas = document.getElementById("canvas");
canvas.addEventListener("mousedown", setLastCoords);
canvas.addEventListener("mousemove", freeForm);

const context = canvas.getContext("2d");

// else the background will be transparant in the base64 png at toDataURL
context.fillStyle = 'white';
context.fillRect(0, 0, canvas.width, canvas.height);

function setLastCoords(e) {
    const { x, y } = canvas.getBoundingClientRect();
    lastX = e.clientX - x;
    lastY = e.clientY - y;
}

function freeForm(e) {
    if (e.buttons !== 1) return; // left button is not pushed yet
    penTool(e);
}

function penTool(e) {
    const { x, y } = canvas.getBoundingClientRect();
    const newX = e.clientX - x;
    const newY = e.clientY - y;

    context.beginPath();
    context.lineWidth = 3;
    context.moveTo(lastX, lastY);
    context.lineTo(newX, newY);
    context.strokeStyle = "black";
    context.stroke();
    context.closePath();

    lastX = newX;
    lastY = newY;
}

let lastX = 0;
let lastY = 0;  