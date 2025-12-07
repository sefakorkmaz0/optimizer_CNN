package layers;

import engine.Tensor;
import java.util.ArrayList;
import java.util.List;

public class MaxPool2D implements Layer {
    private final int poolSize = 2;
    private final int stride = 2;

    // To store indices of max values for backward pass
    // We can store a Tensor of the same shape as output, containing the flat index
    // of the input max.
    private Tensor maxIndices;
    private Tensor lastInputShape; // To know input size during backward

    @Override
    public Tensor forward(Tensor input) {
        // Input: [D, H, W]
        int inD = input.depth;
        int inH = input.height;
        int inW = input.width;

        int outH = (inH - poolSize) / stride + 1;
        int outW = (inW - poolSize) / stride + 1;

        Tensor output = new Tensor(inD, outH, outW);
        this.maxIndices = new Tensor(inD, outH, outW);
        this.lastInputShape = input; // Just to keep reference for dims, or store dims separately.

        for (int d = 0; d < inD; d++) {
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {

                    double maxVal = Double.NEGATIVE_INFINITY;
                    int maxIdx = -1;

                    for (int ky = 0; ky < poolSize; ky++) {
                        for (int kx = 0; kx < poolSize; kx++) {
                            int inY = y * stride + ky;
                            int inX = x * stride + kx;

                            // Manual index calculation for input
                            int idx = d * (inH * inW) + inY * inW + inX;
                            double val = input.data[idx];

                            if (val > maxVal) {
                                maxVal = val;
                                maxIdx = idx;
                            }
                        }
                    }

                    output.set(d, y, x, maxVal);
                    maxIndices.set(d, y, x, maxIdx);
                }
            }
        }

        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // gradOutput: [D, outH, outW]
        // gradInput: [D, inH, inW] (initialized to 0)
        Tensor gradInput = new Tensor(lastInputShape.depth, lastInputShape.height, lastInputShape.width);

        int outD = gradOutput.depth;
        int outH = gradOutput.height;
        int outW = gradOutput.width;

        for (int d = 0; d < outD; d++) {
            for (int y = 0; y < outH; y++) {
                for (int x = 0; x < outW; x++) {
                    double grad = gradOutput.get(d, y, x);
                    int maxIdx = (int) maxIndices.get(d, y, x);

                    // Add gradient to the max index position
                    gradInput.data[maxIdx] += grad;
                }
            }
        }

        return gradInput;
    }

    @Override
    public List<Tensor> getParameters() {
        return new ArrayList<>();
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>();
    }
}
