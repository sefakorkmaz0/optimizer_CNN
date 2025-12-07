package layers;

import engine.Tensor;
import java.util.List;

public interface Layer {
    /**
     * Computes the output of the layer given the input.
     * Stores input for backward pass if necessary.
     */
    Tensor forward(Tensor input);

    /**
     * Computes the gradient of the loss with respect to the input.
     * Also computes gradients for weights/biases if applicable.
     * 
     * @param gradOutput Gradient of loss w.r.t layer output
     * @return Gradient of loss w.r.t layer input
     */
    Tensor backward(Tensor gradOutput);

    /**
     * Returns a list of trainable parameters (weights, biases).
     * Used by the Optimizer.
     */
    List<Tensor> getParameters();

    /**
     * Returns a list of gradients corresponding to the parameters.
     * Used by the Optimizer.
     */
    List<Tensor> getGradients();
}
