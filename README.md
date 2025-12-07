# Fashion MNIST Web Demo

This is a web-based demo for the CNN Fashion MNIST Classifier.

## How to Run

Because the demo loads the model weights from a JSON file (`model_weights.json`), modern browsers will block this if you just double-click `index.html` (due to CORS security policies).

You need to run a simple local web server.

### Option 1: Using Python (Recommended)
If you have Python installed, run this command in this directory:

```bash
# Python 3
python3 -m http.server
```

Then open your browser and go to:
[http://localhost:8000](http://localhost:8000)

### Option 2: VS Code Live Server
If you use VS Code, you can install the "Live Server" extension, right-click `index.html`, and select "Open with Live Server".

## How to Use
1. Draw a clothing item (like a t-shirt, shoe, or bag) in the black box.
2. Click **Predict**.
3. See the AI's guess!
