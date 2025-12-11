package engine.graph;

import engine.Tensor;
import layers.Layer;
import java.util.ArrayList;
import java.util.List;

/**
 * Represents a node in the computation graph.
 * Wraps a Layer and manages connections to other nodes.
 * 
 * This class is a fundamental building block of the DAG-based
 * computation graph, storing both the layer operation and
 * intermediate tensors for forward/backward passes.
 */
public class Node {
    public final String id;
    public final Layer layer;
    public final List<Node> inputs = new ArrayList<>();
    public final List<Node> outputs = new ArrayList<>();

    /** Cached output tensor from forward pass */
    public Tensor output;

    /** Accumulated gradient from backward pass */
    public Tensor gradient;

    public Node(String id, Layer layer) {
        this.id = id;
        this.layer = layer;
    }

    public void addInput(Node inputNode) {
        this.inputs.add(inputNode);
        inputNode.outputs.add(this);
    }

    @Override
    public String toString() {
        return "Node(" + id + ")";
    }
}
