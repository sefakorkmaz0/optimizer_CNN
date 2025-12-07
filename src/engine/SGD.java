package engine;

import layers.Layer;
import java.util.List;

public class SGD implements Optimizer {
    private final double learningRate;

    public SGD(double learningRate) {
        this.learningRate = learningRate;
    }

    @Override
    public void update(List<Layer> layers) {
        for (Layer layer : layers) {
            List<Tensor> params = layer.getParameters();
            List<Tensor> grads = layer.getGradients();

            if (params.size() != grads.size()) {
                throw new RuntimeException("Parameter and Gradient count mismatch");
            }

            for (int i = 0; i < params.size(); i++) {
                Tensor param = params.get(i);
                Tensor grad = grads.get(i);

                // w = w - lr * grad
                for (int j = 0; j < param.data.length; j++) {
                    param.data[j] -= learningRate * grad.data[j];
                }
            }
        }
    }
}
