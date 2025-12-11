package layers;

import engine.Tensor;
import java.util.List;

/**
 * Interface for all neural network layers.
 * 
 * Each layer must implement forward propagation, backward propagation,
 * and provide access to its trainable parameters and gradients.
 * 
 * Time Complexity varies by layer type:
 * - Dense: O(input_size * output_size)
 * - Conv2D: O(filters * input_depth * kernel^2 * output_area)
 * - MaxPool2D: O(input_size)
 * - ReLU/Softmax: O(input_size)
 */
public interface Layer {

    /**
     * Computes the forward pass of the layer.
     * 
     * @param input The input tensor from the previous layer
     * @return The output tensor after applying this layer's transformation
     */
    Tensor forward(Tensor input);

    /**
     * Computes the backward pass (backpropagation) of the layer.
     * 
     * This method:
     * 1. Computes gradients for trainable parameters (weights, biases)
     * 2. Computes gradient with respect to inputs for upstream layers
     * 
     * @param gradOutput Gradient of loss with respect to this layer's output
     * @return Gradient of loss with respect to this layer's input
     */
    Tensor backward(Tensor gradOutput);

    /**
     * Returns the list of trainable parameters (weights, biases).
     * Used by the Optimizer for weight updates.
     * 
     * @return List of parameter tensors (empty for non-trainable layers)
     */
    List<Tensor> getParameters();

    /**
     * Returns the gradients corresponding to each parameter.
     * Must be in the same order as getParameters().
     * 
     * @return List of gradient tensors
     */
    List<Tensor> getGradients();
}
