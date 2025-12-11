/**
 * Neural Vision - Fashion Classifier
 * Pure JavaScript CNN Inference Engine
 */

// DOM Elements
const canvas = document.getElementById('drawingCanvas');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clearBtn');
const predictBtn = document.getElementById('predictBtn');
const predictedLabel = document.getElementById('predictedLabel');
const resultEmoji = document.getElementById('resultEmoji');
const resultCard = document.getElementById('resultCard');
const confidenceBar = document.getElementById('confidenceBar');
const confidenceValue = document.getElementById('confidenceValue');
const probList = document.getElementById('probList');

// Fashion-MNIST Classes with Emojis
const CLASSES = [
    { name: "T-shirt/Top", emoji: "👕" },
    { name: "Trouser", emoji: "👖" },
    { name: "Pullover", emoji: "🧥" },
    { name: "Dress", emoji: "👗" },
    { name: "Coat", emoji: "🧥" },
    { name: "Sandal", emoji: "🩴" },
    { name: "Shirt", emoji: "👔" },
    { name: "Sneaker", emoji: "👟" },
    { name: "Bag", emoji: "👜" },
    { name: "Ankle Boot", emoji: "🥾" }
];

let isDrawing = false;
let modelWeights = null;

// Initialize Canvas
function initCanvas() {
    ctx.fillStyle = "black";
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    ctx.strokeStyle = "white";
    ctx.lineWidth = 18;
    ctx.lineCap = "round";
    ctx.lineJoin = "round";
}

initCanvas();

// Event Listeners
canvas.addEventListener('mousedown', startDrawing);
canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mouseup', stopDrawing);
canvas.addEventListener('mouseout', stopDrawing);

// Touch support for mobile
canvas.addEventListener('touchstart', (e) => { e.preventDefault(); startDrawing(e.touches[0]); });
canvas.addEventListener('touchmove', (e) => { e.preventDefault(); draw(e.touches[0]); });
canvas.addEventListener('touchend', stopDrawing);

clearBtn.addEventListener('click', clearCanvas);
predictBtn.addEventListener('click', predict);

// Load Model Weights
fetch('model_weights.json')
    .then(response => response.json())
    .then(data => {
        modelWeights = data;
        console.log("✅ Model weights loaded successfully");
    })
    .catch(err => {
        console.error("❌ Error loading weights:", err);
        predictedLabel.textContent = "Model load error";
    });

// Drawing Functions
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
    initCanvas();
    predictedLabel.textContent = "Draw something...";
    resultEmoji.textContent = "✏️";
    confidenceBar.style.width = "0%";
    confidenceValue.textContent = "0%";
    probList.innerHTML = "";
    resultCard.classList.remove('active');
}

// Get Input Tensor from Canvas
function getInputTensor() {
    // Resize to 28x28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = 28;
    tempCanvas.height = 28;
    const tempCtx = tempCanvas.getContext('2d');

    tempCtx.drawImage(canvas, 0, 0, 28, 28);

    const imageData = tempCtx.getImageData(0, 0, 28, 28);
    const data = imageData.data;

    // Convert to tensor [1, 28, 28] (flattened)
    const tensor = new Float32Array(28 * 28);
    for (let i = 0; i < 28 * 28; i++) {
        tensor[i] = data[i * 4] / 255.0;
    }
    return tensor;
}

// ============================================
// Neural Network Inference Engine
// ============================================

function predict() {
    if (!modelWeights) {
        alert("Model is still loading...");
        return;
    }

    // Add loading state
    predictBtn.classList.add('loading');
    predictedLabel.textContent = "Analyzing...";

    // Small delay for UX
    setTimeout(() => {
        const input = getInputTensor();

        // Forward Pass
        // 1. Conv2D: 1x28x28 → 8x26x26
        const convOut = conv2d(input, modelWeights.conv2d.weights, modelWeights.conv2d.biases);

        // 2. ReLU
        const reluOut = relu(convOut);

        // 3. MaxPool2D: 8x26x26 → 8x13x13
        const poolOut = maxPool2d(reluOut, 8, 26, 26);

        // 4. Dense: 1352 → 10
        const denseOut = dense(poolOut, modelWeights.dense.weights, modelWeights.dense.biases);

        // 5. Softmax
        const probs = softmax(denseOut);

        displayResults(probs);
        predictBtn.classList.remove('loading');
    }, 100);
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

// ============================================
// Display Results
// ============================================

function displayResults(probs) {
    // Find top prediction
    let maxProb = -1;
    let maxIdx = -1;
    for (let i = 0; i < probs.length; i++) {
        if (probs[i] > maxProb) {
            maxProb = probs[i];
            maxIdx = i;
        }
    }

    // Update main result
    const prediction = CLASSES[maxIdx];
    predictedLabel.textContent = prediction.name;
    resultEmoji.textContent = prediction.emoji;

    const confidence = Math.round(maxProb * 100);
    confidenceBar.style.width = `${confidence}%`;
    confidenceValue.textContent = `${confidence}%`;

    // Animate result card
    resultCard.classList.add('active');

    // Build probability list
    probList.innerHTML = "";
    const sortedIndices = probs
        .map((p, i) => [p, i])
        .sort((a, b) => b[0] - a[0]);

    sortedIndices.forEach(([prob, idx], rank) => {
        const li = document.createElement('li');
        const item = CLASSES[idx];
        const percentage = (prob * 100).toFixed(1);

        li.innerHTML = `
            <span>${item.emoji} ${item.name}</span>
            <span>${percentage}%</span>
        `;

        if (rank === 0) {
            li.classList.add('top');
        }

        probList.appendChild(li);
    });
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    if (e.key === 'c' || e.key === 'C') {
        clearCanvas();
    }
    if (e.key === 'Enter') {
        predict();
    }
});

console.log("🧠 Neural Vision initialized");
console.log("💡 Tip: Press 'C' to clear, 'Enter' to predict");
