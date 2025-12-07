package engine;

import layers.Layer;
import java.util.List;

public interface Optimizer {
    /**
     * Updates the weights of the given layers based on their gradients.
     * 
     * @param layers List of layers in the network.
     */
    void update(List<Layer> layers);
}
