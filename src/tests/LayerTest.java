package tests;

import engine.Tensor;
import layers.Conv2D;
import layers.Dense;
import layers.MaxPool2D;

public class LayerTest {
    public static void main(String[] args) {
        System.out.println("Running Layer Tests...");

        testDense();
        testConv2D();
        testMaxPool2D();

        System.out.println("All Layer Tests Passed!");
    }

    private static void testDense() {
        System.out.print("Testing Dense Layer... ");
        // Input: 2 features
        Tensor input = new Tensor(1, 1, 2);
        input.data[0] = 1.0;
        input.data[1] = 2.0;

        // Dense: 2 input -> 1 output
        Dense dense = new Dense(2, 1);
        // Manually set weights for predictable output
        // Weights: [1, 1, 2] -> w1=0.5, w2=0.5
        dense.getParameters().get(0).fill(0.5);
        dense.getParameters().get(1).fill(0.0); // Bias = 0

        // Forward: 1*0.5 + 2*0.5 = 1.5
        Tensor output = dense.forward(input);
        if (Math.abs(output.data[0] - 1.5) > 1e-6) {
            throw new RuntimeException("Dense Forward failed. Expected 1.5, got " + output.data[0]);
        }

        // Backward
        Tensor gradOutput = new Tensor(1, 1, 1);
        gradOutput.data[0] = 1.0;
        Tensor gradInput = dense.backward(gradOutput);

        // dL/dx = gradOutput * W = 1.0 * 0.5 = 0.5
        if (Math.abs(gradInput.data[0] - 0.5) > 1e-6)
            throw new RuntimeException("Dense Backward failed");

        System.out.println("OK");
    }

    private static void testConv2D() {
        System.out.print("Testing Conv2D Layer... ");
        // Input: 1x3x3
        Tensor input = new Tensor(1, 3, 3);
        input.fill(1.0);

        // Conv: 1 filter, 2x2 kernel
        Conv2D conv = new Conv2D(1, 1, 2);
        conv.getParameters().get(0).fill(1.0); // Kernel all 1s
        conv.getParameters().get(1).fill(0.0); // Bias 0

        // Forward: 3x3 input, 2x2 kernel -> 2x2 output
        // Each output pixel = sum(2x2 window of 1s) * 1 = 4.0
        Tensor output = conv.forward(input);

        if (output.height != 2 || output.width != 2)
            throw new RuntimeException("Conv Output shape mismatch");
        if (Math.abs(output.data[0] - 4.0) > 1e-6)
            throw new RuntimeException("Conv Forward failed. Expected 4.0, got " + output.data[0]);

        System.out.println("OK");
    }

    private static void testMaxPool2D() {
        System.out.print("Testing MaxPool2D Layer... ");
        // Input: 1x4x4
        Tensor input = new Tensor(1, 4, 4);
        input.fill(0.0);
        // Set a max value at (0,0) and (1,1) in the first 2x2 block
        input.set(0, 0, 0, 5.0);
        input.set(0, 1, 1, 3.0);

        MaxPool2D pool = new MaxPool2D();
        Tensor output = pool.forward(input);

        // Output should be 2x2
        // Top-left block max is 5.0
        if (Math.abs(output.get(0, 0, 0) - 5.0) > 1e-6)
            throw new RuntimeException("MaxPool Forward failed");

        // Backward
        Tensor gradOutput = new Tensor(1, 2, 2);
        gradOutput.fill(1.0);
        Tensor gradInput = pool.backward(gradOutput);

        // Gradient should flow only to the max index (0,0)
        if (Math.abs(gradInput.get(0, 0, 0) - 1.0) > 1e-6)
            throw new RuntimeException("MaxPool Backward failed (target)");
        if (Math.abs(gradInput.get(0, 1, 1) - 0.0) > 1e-6)
            throw new RuntimeException("MaxPool Backward failed (non-target)");

        System.out.println("OK");
    }
}
