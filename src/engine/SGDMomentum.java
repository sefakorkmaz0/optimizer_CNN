package engine;

import layers.Layer;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

public class SGDMomentum implements Optimizer {
    private final double learningRate;
    private final double momentum;

    // Store velocity for each parameter tensor
    // Map<Tensor (Param), double[] (Velocity)>
    private final Map<Tensor, double[]> velocities;

    public SGDMomentum(double learningRate, double momentum) {
        this.learningRate = learningRate;
        this.momentum = momentum;
        this.velocities = new HashMap<>();
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer : layers) {
            List<Tensor> params = layer.getParameters();
            List<Tensor> grads = layer.getGradients();

            for (int i = 0; i < params.size(); i++) {
                Tensor param = params.get(i);
                Tensor grad = grads.get(i);

                // Initialize velocity if not exists
                if (!velocities.containsKey(param)) {
                    velocities.put(param, new double[param.data.length]);
                }
                double[] v = velocities.get(param);

                // v = momentum * v - lr * grad
                // w = w + v
                for (int j = 0; j < param.data.length; j++) {
                    v[j] = momentum * v[j] - learningRate * grad.data[j];
                    param.data[j] += v[j];
                }
            }
        }
    }
}
