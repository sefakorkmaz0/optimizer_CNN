package engine.graph;

import engine.Tensor;
import java.util.*;

/**
 * Manages the computation graph and execution order.
 * Uses Topological Sort to determine the correct sequence of operations.
 */
public class ComputationGraph {
    private List<Node> nodes = new ArrayList<>();
    private List<Node> executionOrder = new ArrayList<>();

    // For simple sequential-like access (Input Node -> Output Node)
    // In a general DAG, we might have multiple inputs/outputs.
    // Here we assume a single entry for simplicity of the API, or user sets inputs
    // manually.

    public void addNode(Node node) {
        nodes.add(node);
    }

    /**
     * Compiles the graph by performing Topological Sort.
     * This must be called before forward().
     */
    public void compile() {
        executionOrder.clear();

        // Kahns Algorithm for Topological Sort
        Map<Node, Integer> inDegree = new HashMap<>();
        Queue<Node> queue = new LinkedList<>();

        // Calculate in-degrees
        for (Node node : nodes) {
            inDegree.put(node, 0);
        }

        for (Node node : nodes) {
            for (Node out : node.outputs) {
                inDegree.put(out, inDegree.getOrDefault(out, 0) + 1);
            }
        }

        // Add nodes with 0 in-degree to queue
        for (Node node : nodes) {
            if (inDegree.get(node) == 0) {
                queue.add(node);
            }
        }

        while (!queue.isEmpty()) {
            Node u = queue.poll();
            executionOrder.add(u);

            for (Node v : u.outputs) {
                inDegree.put(v, inDegree.get(v) - 1);
                if (inDegree.get(v) == 0) {
                    queue.add(v);
                }
            }
        }

        if (executionOrder.size() != nodes.size()) {
            throw new RuntimeException("Cycle detected in graph! Topological sort failed.");
        }

        System.out.println("Graph Compiled. Execution Order: " + executionOrder);
    }

    /**
     * Executes the graph forward pass.
     * 
     * Time Complexity: O(V + E) where V is nodes and E is edges
     * 
     * @param input The input tensor for the first node in topological order.
     * @return The output tensor from the last node.
     */
    public Tensor forward(Tensor input) {
        if (executionOrder.isEmpty()) {
            throw new RuntimeException("Graph not compiled! Call compile() first.");
        }

        // Feed input to the first node (placeholder or first layer)
        Node inputNode = executionOrder.get(0);
        inputNode.output = (inputNode.layer != null)
                ? inputNode.layer.forward(input)
                : input;

        // Execute remaining nodes in topological order
        for (int i = 1; i < executionOrder.size(); i++) {
            Node node = executionOrder.get(i);

            // Collect input tensor from parent nodes
            Tensor nodeInput;
            if (node.inputs.size() == 1) {
                nodeInput = node.inputs.get(0).output;
            } else {
                // Multiple inputs: merge by summation
                nodeInput = node.inputs.get(0).output.copy();
                for (int j = 1; j < node.inputs.size(); j++) {
                    nodeInput.add(node.inputs.get(j).output);
                }
            }

            // Execute layer forward pass
            node.output = node.layer.forward(nodeInput);
        }

        return executionOrder.get(executionOrder.size() - 1).output;
    }

    public void backward(Tensor gradOutput) {
        if (executionOrder.isEmpty()) {
            throw new RuntimeException("Graph not compiled!");
        }

        // Initialize backward pass
        // Reset gradients for all nodes
        for (Node n : nodes) {
            n.gradient = null;
        }

        // Set the gradient for the last node (Output Layer)
        Node lastNode = executionOrder.get(executionOrder.size() - 1);
        lastNode.gradient = gradOutput;

        // Iterate in reverse topological order
        for (int i = executionOrder.size() - 1; i >= 0; i--) {
            Node node = executionOrder.get(i);

            // If this node has no gradient accumulated (e.g. unused branch), skip
            if (node.gradient == null)
                continue;

            // If it's a placeholder input node (no layer), nothing to backprop through
            if (node.layer == null)
                continue;

            // Backpropagate through the layer:
            // Input: dL/dOutput (node.gradient)
            // Output: dL/dInput (inputGrad)
            Tensor inputGrad = node.layer.backward(node.gradient);

            // Distribute this gradient to parents (inputs of this node)
            if (node.inputs.isEmpty())
                continue;

            if (node.inputs.size() == 1) {
                Node parent = node.inputs.get(0);
                if (parent.gradient == null) {
                    parent.gradient = inputGrad;
                } else {
                    parent.gradient.add(inputGrad);
                }
            } else {
                // If node has multiple inputs (e.g. Sum node), the gradient flows to all.
                // d(Sum)/dx = 1, so dL/dx = dL/dSum * 1
                for (Node parent : node.inputs) {
                    if (parent.gradient == null) {
                        parent.gradient = inputGrad.copy();
                    } else {
                        parent.gradient.add(inputGrad);
                    }
                }
            }
        }
    }

    public List<Node> getExecutionOrder() {
        return executionOrder;
    }
}
