package layers;

import engine.Tensor;
import java.util.ArrayList;
import java.util.List;

public class Softmax implements Layer {
    private Tensor output; // Store for backward pass

    @Override
    public Tensor forward(Tensor input) {
        // Input: [1, 1, Classes] (Usually flattened)
        // We assume 1D input effectively.

        Tensor out = new Tensor(input.depth, input.height, input.width);

        double max = Double.NEGATIVE_INFINITY;
        for (double val : input.data) {
            if (val > max)
                max = val;
        }

        double sum = 0;
        for (int i = 0; i < input.data.length; i++) {
            out.data[i] = Math.exp(input.data[i] - max); // Stable softmax
            sum += out.data[i];
        }

        for (int i = 0; i < input.data.length; i++) {
            out.data[i] /= sum;
        }

        this.output = out;
        return out;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        // Jacobian of Softmax is complex.
        // Usually Softmax is combined with CrossEntropy.
        // If we implement them separately:
        // dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
        // dy_j/dx_i = y_i * (delta_ij - y_j)

        // However, standard practice is to pass (Prediction - Target) directly if using
        // CrossEntropy.
        // But here we need to stick to the Layer interface.
        // Let's implement the full Jacobian multiplication for correctness if
        // separated.

        Tensor gradInput = new Tensor(output.depth, output.height, output.width);

        int len = output.data.length;
        for (int i = 0; i < len; i++) {
            double sum = 0;
            for (int j = 0; j < len; j++) {
                double kron = (i == j) ? 1.0 : 0.0;
                double s_i = output.data[i];
                double s_j = output.data[j];

                // derivative of s_j w.r.t x_i
                double dSj_dXi = s_j * (kron - s_i); // Wait, formula is s_i * (kron - s_j) if we look at dSi/dXj

                // Let's use: dL/dx_i = sum_j (dL/dy_j * dy_j/dx_i)
                // dy_j/dx_i = y_j * (delta_ij - y_i)

                double grad_j = gradOutput.data[j];
                sum += grad_j * (s_j * (kron - s_i));
            }
            gradInput.data[i] = sum;
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
