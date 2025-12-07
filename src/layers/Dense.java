package layers;

import engine.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Dense implements Layer {
    private Tensor weights; // Shape: [1, inputSize, outputSize] or similar. Let's use [1, outputSize,
                            // inputSize] for easier dot product?
                            // Actually, standard is W * x + b. If x is col vector.
                            // Here Tensor is 1D. Let's say input is size I, output is size O.
                            // Weights size: O x I.
    private Tensor biases; // Size: O
    private Tensor lastInput;

    private Tensor gradWeights;
    private Tensor gradBiases;

    private final int inputSize;
    private final int outputSize;

    public Dense(int inputSize, int outputSize) {
        this.inputSize = inputSize;
        this.outputSize = outputSize;

        // Weights: [1, outputSize, inputSize]
        // We use 3D tensor but treat it as 2D matrix [Output, Input]
        this.weights = new Tensor(1, outputSize, inputSize);
        this.biases = new Tensor(1, 1, outputSize);

        this.gradWeights = new Tensor(1, outputSize, inputSize);
        this.gradBiases = new Tensor(1, 1, outputSize);

        initializeParams();
    }

    private void initializeParams() {
        // He Initialization
        double std = Math.sqrt(2.0 / inputSize);
        Random rand = new Random();
        for (int i = 0; i < weights.data.length; i++) {
            weights.data[i] = rand.nextGaussian() * std;
        }
        // Biases 0
    }

    @Override
    public Tensor forward(Tensor input) {
        // Input might be [D, H, W]. Flatten it conceptually.
        // Check size
        if (input.data.length != inputSize) {
            throw new IllegalArgumentException(
                    "Input size mismatch. Expected " + inputSize + ", got " + input.data.length);
        }

        this.lastInput = input;
        Tensor output = new Tensor(1, 1, outputSize);

        // y = Wx + b
        for (int o = 0; o < outputSize; o++) {
            double sum = 0;
            for (int i = 0; i < inputSize; i++) {
                // weights[o][i] * input[i]
                sum += weights.get(0, o, i) * input.data[i];
            }
            sum += biases.data[o];
            output.data[o] = sum;
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // gradOutput is dL/dy (size O)
        // We need:
        // dL/dW = dL/dy * dy/dW = gradOutput * input^T
        // dL/db = gradOutput
        // dL/dx = dL/dy * dy/dx = gradOutput^T * W

        Tensor gradInput = new Tensor(lastInput.depth, lastInput.height, lastInput.width); // Shape of original input

        // Compute Gradients for Weights and Biases
        for (int o = 0; o < outputSize; o++) {
            double grad = gradOutput.data[o];
            gradBiases.data[o] += grad; // Accumulate? Usually we overwrite or accumulate if batch.
                                        // For simple SGD per sample, overwrite. But if batch, accumulate.
                                        // Let's assume overwrite for this step, optimizer handles update.
                                        // Actually, standard backprop sets the gradient for the current step.
            gradBiases.data[o] = grad;

            for (int i = 0; i < inputSize; i++) {
                // dL/dW_oi = grad_o * input_i
                gradWeights.set(0, o, i, grad * lastInput.data[i]);

                // Accumulate gradient for input
                // dL/dx_i += grad_o * W_oi
                gradInput.data[i] += grad * weights.get(0, o, i);
            }
        }

        return gradInput;
    }

    @Override
    public List<Tensor> getParameters() {
        List<Tensor> params = new ArrayList<>();
        params.add(weights);
        params.add(biases);
        return params;
    }

    @Override
    public List<Tensor> getGradients() {
        List<Tensor> grads = new ArrayList<>();
        grads.add(gradWeights);
        grads.add(gradBiases);
        return grads;
    }
}
