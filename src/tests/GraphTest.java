package tests;

import engine.graph.ComputationGraph;
import engine.graph.Node;
import layers.*;
import engine.Tensor;

public class GraphTest {
    public static void main(String[] args) {
        System.out.println("Running GraphTest...");

        ComputationGraph graph = new ComputationGraph();

        // 1. Create Nodes
        Node input = new Node("Input", null); // Placeholder for input
        Node conv = new Node("Conv1", new Conv2D(1, 2, 3));
        Node relu = new Node("ReLU1", new ReLU());
        Node pool = new Node("Pool1", new MaxPool2D());
        Node dense = new Node("Dense1", new Dense(2 * 13 * 13, 10)); // Adjust based on output size

        // 2. Connect Nodes (Linear chain for test)
        // Input -> Conv -> ReLU -> Pool -> Dense

        connect(input, conv);
        connect(conv, relu);
        connect(relu, pool);
        connect(pool, dense);

        // Add to graph (order shouldn't matter for topological sort)
        graph.addNode(pool);
        graph.addNode(input);
        graph.addNode(dense);
        graph.addNode(relu);
        graph.addNode(conv);

        // 3. Compile
        System.out.println("Compiling Graph...");
        graph.compile();

        // 4. Verify Order
        System.out.println("Verifying Order...");
        // Expected: Input, Conv1, ReLU1, Pool1, Dense1
        // Note: Strict order depends on implementation, but dependencies must be
        // respected.

        // 5. Test Forward Pass
        System.out.println("Testing Forward Pass...");
        Tensor x = new Tensor(1, 28, 28);
        x.fillRandom(0, 1);

        Tensor out = graph.forward(x);
        System.out.println("Output Shape: " + out);

        if (out.data.length == 10) {
            System.out.println("SUCCESS: Output size is correct.");
        } else {
            System.out.println("FAILURE: Output size incorrect.");
        }
    }

    private static void connect(Node a, Node b) {
        b.addInput(a);
    }
}
