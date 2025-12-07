const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictedLabel = document.getElementById('predictedLabel');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const probList = document.getElementById('probList');

const CLASSES = [
    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
];

let isDrawing = false;
let modelWeights = null;

// Initialize Canvas
ctx.fillStyle = "black";
ctx.fillRect(0, 0, canvas.width, canvas.height);
ctx.strokeStyle = "white";
ctx.lineWidth = 15;
ctx.lineCap = "round";
ctx.lineJoin = "round";

// Event Listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);
// Touch support
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e.touches[0]); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e.touches[0]); });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', predict);

// Load Model
fetch('model_weights.json')
    .then(response => response.json())
    .then(data => {
        modelWeights = data;
        console.log("Model weights loaded");
    })
    .catch(err => console.error("Error loading weights:", err));

function startDrawing(e) {
    isDrawing = true;
    draw(e);
}

function draw(e) {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const x = (e.clientX || e.pageX) - rect.left;
    const y = (e.clientY || e.pageY) - rect.top;

    ctx.lineTo(x, y);
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(x, y);
}

function stopDrawing() {
    isDrawing = false;
    ctx.beginPath();
}

function clearCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    predictedLabel.innerText = "...";
    confidenceBar.style.width = "0%";
    confidenceValue.innerText = "0%";
    probList.innerHTML = "";
}

function getInputTensor() {
    // 1. Resize to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    // Draw main canvas onto temp canvas (scaling down)
    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    // 2. Get pixel data
    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data; // RGBA

    // 3. Convert to tensor [1, 28, 28] (flattened)
    const tensor = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        // Take Red channel (since it's grayscale/BW) and normalize 0-1
        tensor[i] = data[i * 4] / 255.0;
    }
    return tensor;
}

// --- Inference Engine ---

function predict() {
    if (!modelWeights) {
        alert("Model loading...");
        return;
    }

    const input = getInputTensor(); // [784]

    // 1. Conv2D
    // Input: 1x28x28, Weights: [8, 1, 9], Biases: [8]
    // Output: 8x26x26
    const convOut = conv2d(input, modelWeights.conv2d.weights, modelWeights.conv2d.biases);

    // 2. ReLU
    const reluOut = relu(convOut);

    // 3. MaxPool2D
    // Input: 8x26x26 -> Output: 8x13x13
    const poolOut = maxPool2d(reluOut, 8, 26, 26);

    // 4. Dense
    // Input: 1352, Weights: [10, 1352], Biases: [10] (Wait, Java export format needs check)
    // My Java export flattens everything. 
    // Dense weights in Java are [10, 1352] flattened? No, Java Tensor is [Depth, Height, Width].
    // Dense weights in Java: new Tensor(outputSize, inputSize, 1) -> [10, 1352, 1]
    const denseOut = dense(poolOut, modelWeights.dense.weights, modelWeights.dense.biases);

    // 5. Softmax
    const probs = softmax(denseOut);

    displayResults(probs);
}

function conv2d(input, weights, biases) {
    const inH = 28, inW = 28;
    const kSize = 3;
    const numFilters = 8;
    const outH = 26, outW = 26;

    const output = new Float32Array(numFilters * outH * outW);

    for (let f = 0; f < numFilters; f++) {
        const bias = biases[f];
        for (let y = 0; y < outH; y++) {
            for (let x = 0; x < outW; x++) {
                let sum = 0;
                for (let ky = 0; ky < kSize; ky++) {
                    for (let kx = 0; kx < kSize; kx++) {
                        const inVal = input[(y + ky) * inW + (x + kx)];
                        // Weights: [numFilters, 1, 9] -> flattened
                        // Index: f * 9 + ky * 3 + kx
                        const wVal = weights[f * 9 + ky * 3 + kx];
                        sum += inVal * wVal;
                    }
                }
                output[f * (outH * outW) + y * outW + x] = sum + bias;
            }
        }
    }
    return output;
}

function relu(input) {
    return input.map(x => Math.max(0, x));
}

function maxPool2d(input, depth, height, width) {
    const outH = height / 2;
    const outW = width / 2;
    const output = new Float32Array(depth * outH * outW);

    for (let d = 0; d < depth; d++) {
        for (let y = 0; y < outH; y++) {
            for (let x = 0; x < outW; x++) {
                let maxVal = -Infinity;
                // 2x2 window
                for (let ky = 0; ky < 2; ky++) {
                    for (let kx = 0; kx < 2; kx++) {
                        const val = input[d * (height * width) + (y * 2 + ky) * width + (x * 2 + kx)];
                        if (val > maxVal) maxVal = val;
                    }
                }
                output[d * (outH * outW) + y * outW + x] = maxVal;
            }
        }
    }
    return output;
}

function dense(input, weights, biases) {
    const inputSize = 1352; // 8 * 13 * 13
    const outputSize = 10;
    const output = new Float32Array(outputSize);

    for (let i = 0; i < outputSize; i++) {
        let sum = 0;
        for (let j = 0; j < inputSize; j++) {
            // Weights: [10, 1352] -> flattened
            // Index: i * 1352 + j
            sum += input[j] * weights[i * inputSize + j];
        }
        output[i] = sum + biases[i];
    }
    return output;
}

function softmax(input) {
    const max = Math.max(...input);
    const exps = input.map(x => Math.exp(x - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return exps.map(x => x / sum);
}

function displayResults(probs) {
    // Find max
    let maxProb = -1;
    let maxIdx = -1;
    for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIdx = i;
        }
    }

    // Update UI
    predictedLabel.innerText = CLASSES[maxIdx];
    const confidence = Math.round(maxProb * 100);
    confidenceBar.style.width = `${confidence}%`;
    confidenceValue.innerText = `${confidence}%`;

    // List all
    probList.innerHTML = "";
    const sortedIndices = probs.map((p, i) => [p, i])
        .sort((a, b) => b[0] - a[0]);

    sortedIndices.forEach(([prob, idx]) => {
        const li = document.createElement('li');
        li.innerHTML = `<span>${CLASSES[idx]}</span> <span>${(prob * 100).toFixed(1)}%</span>`;
        if (idx === maxIdx) li.style.fontWeight = "bold";
        probList.appendChild(li);
    });
}
