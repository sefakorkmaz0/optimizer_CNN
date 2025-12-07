package layers;

import engine.Tensor;
import java.util.ArrayList;
import java.util.List;
import java.util.Random;

public class Conv2D implements Layer {
    private Tensor weights; // [numFilters, inputDepth, kernelSize*kernelSize]
    private Tensor biases; // [1, 1, numFilters] -> treating as 1D array of size numFilters
    private Tensor lastInput;

    private Tensor gradWeights;
    private Tensor gradBiases;

    private final int numFilters;
    private final int kernelSize;
    private final int inputDepth;
    private final int stride = 1; // Fixed for now

    public Conv2D(int inputDepth, int numFilters, int kernelSize) {
        this.inputDepth = inputDepth;
        this.numFilters = numFilters;
        this.kernelSize = kernelSize;

        // Weights: [numFilters, inputDepth, K*K]
        // This mapping allows us to access filter 'f', channel 'c' easily.
        this.weights = new Tensor(numFilters, inputDepth, kernelSize * kernelSize);
        this.biases = new Tensor(1, 1, numFilters);

        this.gradWeights = new Tensor(numFilters, inputDepth, kernelSize * kernelSize);
        this.gradBiases = new Tensor(1, 1, numFilters);

        initializeParams();
    }

    private void initializeParams() {
        // He Initialization
        double n = inputDepth * kernelSize * kernelSize;
        double std = Math.sqrt(2.0 / n);
        Random rand = new Random();
        for (int i = 0; i < weights.data.length; i++) {
            weights.data[i] = rand.nextGaussian() * std;
        }
    }

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;

        int inH = input.height;
        int inW = input.width;

        int outH = (inH - kernelSize) / stride + 1;
        int outW = (inW - kernelSize) / stride + 1;

        if (outH <= 0 || outW <= 0) {
            throw new IllegalArgumentException("Input too small for kernel size");
        }

        Tensor output = new Tensor(numFilters, outH, outW);

        // Naive 6-loop implementation (Batch=1)
        // For each filter
        for (int f = 0; f < numFilters; f++) {
            double bias = biases.data[f];

            // For each output position
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {

                    double sum = 0;

                    // Convolve over input depth and kernel window
                    for (int c = 0; c < inputDepth; c++) {
                        for (int ky = 0; ky < kernelSize; ky++) {
                            for (int kx = 0; kx < kernelSize; kx++) {
                                int inY = y * stride + ky;
                                int inX = x * stride + kx;

                                double val = input.get(c, inY, inX);
                                double w = weights.get(f, c, ky * kernelSize + kx);

                                sum += val * w;
                            }
                        }
                    }

                    output.set(f, y, x, sum + bias);
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // gradOutput: [numFilters, outH, outW]

        Tensor gradInput = new Tensor(inputDepth, lastInput.height, lastInput.width);

        int outH = gradOutput.height;
        int outW = gradOutput.width;

        // Zero out gradients first (created with 0s by default)
        gradWeights.fill(0);
        gradBiases.fill(0);

        // Loop over gradOutput
        for (int f = 0; f < numFilters; f++) {
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    double chainGrad = gradOutput.get(f, y, x);

                    // Accumulate bias gradient
                    gradBiases.data[f] += chainGrad;

                    // Loop over kernel to update weights and input gradients
                    for (int c = 0; c < inputDepth; c++) {
                        for (int ky = 0; ky < kernelSize; ky++) {
                            for (int kx = 0; kx < kernelSize; kx++) {
                                int inY = y * stride + ky;
                                int inX = x * stride + kx;

                                // dL/dW = chainGrad * input
                                double inputVal = lastInput.get(c, inY, inX);
                                int wIdx = weights.getIndex(f, c, ky * kernelSize + kx);
                                gradWeights.data[wIdx] += chainGrad * inputVal;

                                // dL/dx = chainGrad * weight
                                double wVal = weights.data[wIdx];
                                // Accumulate to gradInput
                                // Note: gradInput is shared across filters, so we use +=
                                int inIdx = gradInput.getIndex(c, inY, inX);
                                gradInput.data[inIdx] += chainGrad * wVal;
                            }
                        }
                    }
                }
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
