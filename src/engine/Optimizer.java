package engine;

import layers.Layer;
import java.util.List;

/**
 * Interface for optimization algorithms.
 * 
 * Optimizers update the trainable parameters of layers based on
 * computed gradients. Different optimizers vary in:
 * - Convergence speed
 * - Memory requirements
 * - Robustness to hyperparameters
 * 
 * Implementations:
 * - SGD: Simple gradient descent, O(W) space
 * - SGDMomentum: Adds velocity tracking, O(2W) space
 * - Adam: Adaptive moments, O(3W) space
 * 
 * where W is the total number of trainable parameters.
 */
public interface Optimizer {

    /**
     * Updates the parameters of all layers based on their gradients.
     * 
     * Time Complexity: O(W) where W is total parameters
     * 
     * @param layers List of layers containing trainable parameters
     */
    void update(List<Layer> layers);
}
