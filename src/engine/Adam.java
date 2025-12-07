package engine;

import layers.Layer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class Adam implements Optimizer {
    private final double learningRate;
    private final double beta1 = 0.9;
    private final double beta2 = 0.999;
    private final double epsilon = 1e-8;

    private int t = 0; // Time step

    // Maps for m (1st moment) and v (2nd moment)
    private final Map<Tensor, double[]> m;
    private final Map<Tensor, double[]> v;

    public Adam(double learningRate) {
        this.learningRate = learningRate;
        this.m = new HashMap<>();
        this.v = new HashMap<>();
    }

    @Override
    public void update(List<Layer> layers) {
        t++;
        for (Layer layer : layers) {
            List<Tensor> params = layer.getParameters();
            List<Tensor> grads = layer.getGradients();

            for (int i = 0; i < params.size(); i++) {
                Tensor param = params.get(i);
                Tensor grad = grads.get(i);

                if (!m.containsKey(param)) {
                    m.put(param, new double[param.data.length]);
                    v.put(param, new double[param.data.length]);
                }

                double[] mVec = m.get(param);
                double[] vVec = v.get(param);

                for (int j = 0; j < param.data.length; j++) {
                    double g = grad.data[j];

                    // Update biased first moment estimate
                    mVec[j] = beta1 * mVec[j] + (1 - beta1) * g;

                    // Update biased second raw moment estimate
                    vVec[j] = beta2 * vVec[j] + (1 - beta2) * g * g;

                    // Compute bias-corrected first moment estimate
                    double mHat = mVec[j] / (1 - Math.pow(beta1, t));

                    // Compute bias-corrected second raw moment estimate
                    double vHat = vVec[j] / (1 - Math.pow(beta2, t));

                    // Update parameters
                    param.data[j] -= learningRate * mHat / (Math.sqrt(vHat) + epsilon);
                }
            }
        }
    }
}
