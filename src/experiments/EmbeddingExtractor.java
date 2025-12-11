package experiments;

import data.DataLoader;
import data.DataLoader.DataPoint;
import engine.*;
import engine.graph.*;
import layers.*;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.List;

/**
 * Extracts embedding vectors from trained models for visualization.
 * Trains separate models with SGD and Adam, then saves the Dense layer
 * outputs (logits) to CSV files for t-SNE analysis.
 */
public class EmbeddingExtractor {
    private static final int EPOCHS = 3;
    private static final String TRAIN_FILE = "data/fashion-mnist_train.csv";
    private static final String TEST_FILE = "data/fashion-mnist_test.csv";

    public static void main(String[] args) throws IOException {
        int n = 60000; // Use full dataset for best quality

        System.out.println("Loading Data...");
        List<DataPoint> trainData = DataLoader.load(TRAIN_FILE, n);
        List<DataPoint> testData = DataLoader.load(TEST_FILE, 1000);

        // 1. Train and Extract SGD
        System.out.println("\n--- Processing SGD ---");
        ComputationGraph graphSGD = buildModel();
        train(graphSGD, new SGD(0.01), trainData, "SGD");
        extractEmbeddings(graphSGD, testData, "embeddings_SGD.csv");

        // 2. Train and Extract Adam
        System.out.println("\n--- Processing Adam ---");
        ComputationGraph graphAdam = buildModel();
        train(graphAdam, new Adam(0.001), trainData, "Adam");
        extractEmbeddings(graphAdam, testData, "embeddings_Adam.csv");

        System.out.println("\nComparison embeddings generated.");
    }

    private static ComputationGraph buildModel() {
        ComputationGraph graph = new ComputationGraph();
        Node inputNode = new Node("Input", null);
        Node convNode = new Node("Conv1", new Conv2D(1, 8, 3));
        Node reluNode = new Node("ReLU1", new ReLU());
        Node poolNode = new Node("Pool1", new MaxPool2D());
        // Dense Layer (The Embedding Layer) - named specifically to find it later if
        // needed
        Node denseNode = new Node("Dense1", new Dense(8 * 13 * 13, 10));
        Node softmaxNode = new Node("Softmax", new Softmax());

        // Connections (addInput handles bi-directional link)
        convNode.addInput(inputNode);
        reluNode.addInput(convNode);
        poolNode.addInput(reluNode);
        denseNode.addInput(poolNode);
        softmaxNode.addInput(denseNode);

        graph.addNode(inputNode);
        graph.addNode(convNode);
        graph.addNode(reluNode);
        graph.addNode(poolNode);
        graph.addNode(denseNode);
        graph.addNode(softmaxNode);

        graph.compile();
        return graph;
    }

    private static void train(ComputationGraph graph, Optimizer optimizer, List<DataPoint> data, String name) {
        CrossEntropyLoss lossFunc = new CrossEntropyLoss();
        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            DataLoader.shuffle(data);
            double epochLoss = 0;

            for (DataPoint dp : data) {
                Tensor out = graph.forward(dp.input);
                epochLoss += lossFunc.forward(out, dp.label);

                Tensor grad = lossFunc.backward(out, dp.label);
                graph.backward(grad);

                List<Layer> layers = new ArrayList<>();
                for (Node node : graph.getExecutionOrder()) {
                    if (node.layer != null)
                        layers.add(node.layer);
                }
                optimizer.update(layers);
            }
            System.out.printf("[%s] Epoch %d/%d Loss: %.4f Time: %dms\n", name, epoch, EPOCHS, epochLoss / data.size(),
                    (System.currentTimeMillis() - startTime));
        }
    }

    private static void extractEmbeddings(ComputationGraph graph, List<DataPoint> testData, String filename)
            throws IOException {
        System.out.println("Extracting embeddings to " + filename + "...");

        // Find the Dense node
        Node denseNode = null;
        for (Node n : graph.getExecutionOrder()) {
            if ("Dense1".equals(n.id)) {
                denseNode = n;
                break;
            }
        }

        if (denseNode == null)
            throw new RuntimeException("Could not find Dense1 node");

        try (PrintWriter writer = new PrintWriter(new FileWriter("experiments/" + filename))) {
            // Header
            for (int i = 0; i < 10; i++)
                writer.print("d" + i + ",");
            writer.println("label");

            for (DataPoint dp : testData) {
                graph.forward(dp.input);

                // The 'denseNode.output' now contains the logits (embeddings)
                Tensor embed = denseNode.output;

                for (double v : embed.data) {
                    writer.print(v + ",");
                }
                writer.println(dp.labelIndex);
            }
        }
    }
}
