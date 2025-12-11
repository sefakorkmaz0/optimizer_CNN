package experiments;

import data.DataLoader;
import data.DataLoader.DataPoint;
import engine.*;
import engine.graph.*;
import layers.*;
import utils.CircularQueue;

import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

public class ExperimentRunner {
    private static final int EPOCHS = 5;
    private static final int BATCH_SIZE = 32;

    private static final String TRAIN_FILE = "data/fashion-mnist_train.csv";
    private static final String TEST_FILE = "data/fashion-mnist_test.csv";
    private static final String RESULTS_FILE = "experiments/results.csv";

    public static void main(String[] args) throws IOException {
        // Load existing results to support resuming
        java.util.Set<String> completedTrials = new java.util.HashSet<>();
        java.io.File resultsFile = new java.io.File(RESULTS_FILE);
        if (resultsFile.exists()) {
            System.out.println("Found existing results. Checking for completion...");
            try (java.io.BufferedReader br = new java.io.BufferedReader(new java.io.FileReader(resultsFile))) {
                String line;
                br.readLine(); // Skip header
                while ((line = br.readLine()) != null) {
                    String[] parts = line.split(",");
                    if (parts.length >= 4) {
                        String opt = parts[0];
                        String nStr = parts[1];
                        String tStr = parts[2];
                        int epoch = Integer.parseInt(parts[3]);
                        if (epoch == EPOCHS) {
                            completedTrials.add(opt + "_" + nStr + "_" + tStr);
                        }
                    }
                }
            }
        } else {
            // Initialize CSV if not exists
            try (PrintWriter writer = new PrintWriter(new FileWriter(RESULTS_FILE))) {
                writer.println("Optimizer,InputSize,Trial,Epoch,TimeMs,MemoryMB,Accuracy,Loss");
            }
        }

        int[] sizes = { 1000, 10000, 60000 };
        String[] optimizers = { "SGD", "Momentum", "Adam" };
        int trials = 3;

        // Warm-up JVM
        System.out.println("Warming up JVM...");
        runExperiment("SGD", 100, 0, true);

        for (int n : sizes) {
            for (String optName : optimizers) {
                for (int t = 1; t <= trials; t++) {
                    String key = optName + "_" + n + "_" + t;
                    if (completedTrials.contains(key)) {
                        System.out.println("Skipping completed: " + key);
                        continue;
                    }

                    System.out.printf("Running: %s, N=%d, Trial=%d\n", optName, n, t);
                    runExperiment(optName, n, t, false);
                    System.gc(); // Suggest GC between runs
                }
            }
        }
        System.out.println("All experiments completed. Results saved to " + RESULTS_FILE);
    }

    private static void runExperiment(String optName, int n, int trial, boolean isWarmup) throws IOException {
        // Load Data
        List<DataPoint> trainData = DataLoader.load(TRAIN_FILE, n);
        List<DataPoint> testData = DataLoader.load(TEST_FILE, 1000);

        // Build Graph Model
        ComputationGraph graph = new ComputationGraph();

        // Define Nodes
        Node inputNode = new Node("Input", null); // Placeholder for input
        Node convNode = new Node("Conv1", new Conv2D(1, 8, 3));
        Node reluNode = new Node("ReLU1", new ReLU());
        Node poolNode = new Node("Pool1", new MaxPool2D());
        Node denseNode = new Node("Dense1", new Dense(8 * 13 * 13, 10)); // 8 filters * 13x13 size
        Node softmaxNode = new Node("Softmax", new Softmax());

        // Connect Nodes (Linear Chain)
        connect(inputNode, convNode);
        connect(convNode, reluNode);
        connect(reluNode, poolNode);
        connect(poolNode, denseNode);
        connect(denseNode, softmaxNode);

        // Add to Graph
        graph.addNode(inputNode);
        graph.addNode(convNode);
        graph.addNode(reluNode);
        graph.addNode(poolNode);
        graph.addNode(denseNode);
        graph.addNode(softmaxNode);

        // Compile (Topological Sort)
        if (!isWarmup)
            System.out.println("Compiling Computation Graph...");
        graph.compile();

        // Optimizer
        Optimizer optimizer;
        double lr = 0.01;
        switch (optName) {
            case "SGD":
                optimizer = new SGD(lr);
                break;
            case "Momentum":
                optimizer = new SGDMomentum(lr, 0.9);
                break;
            case "Adam":
                optimizer = new Adam(0.001);
                break;
            default:
                throw new IllegalArgumentException("Unknown optimizer");
        }

        CrossEntropyLoss lossFunc = new CrossEntropyLoss();
        CircularQueue lossQueue = new CircularQueue(100);

        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            DataLoader.shuffle(trainData);
            double epochLoss = 0;
            int correct = 0;

            // Training Loop
            for (int i = 0; i < trainData.size(); i++) {
                DataPoint dp = trainData.get(i);

                // Forward via Graph
                Tensor out = graph.forward(dp.input);

                // Loss & Accuracy
                double l = lossFunc.forward(out, dp.label);
                epochLoss += l;
                lossQueue.add(l);

                int pred = argmax(out);
                if (pred == dp.labelIndex)
                    correct++;

                // Backward via Graph
                Tensor grad = lossFunc.backward(out, dp.label);
                graph.backward(grad);

                // Update (Extract layers from graph to pass to optimizer)
                // In a real framework, optimizer would iterate graph nodes.
                // Here we adapt to existing Optimizer API.
                List<Layer> trainableLayers = new ArrayList<>();
                for (Node node : graph.getExecutionOrder()) {
                    if (node.layer != null)
                        trainableLayers.add(node.layer);
                }
                optimizer.update(trainableLayers);
            }

            long currentTime = System.currentTimeMillis();
            long elapsed = currentTime - startTime;
            double acc = (double) correct / trainData.size();
            double avgLoss = epochLoss / trainData.size();
            double memory = (Runtime.getRuntime().totalMemory() - Runtime.getRuntime().freeMemory())
                    / (1024.0 * 1024.0);

            if (!isWarmup) {
                logResult(optName, n, trial, epoch, elapsed, memory, acc, avgLoss);
            }
            if (!isWarmup || epoch % 5 == 0) // Reducing spam during warmup
                System.out.printf("Epoch %d: Loss=%.4f, Acc=%.4f, Time=%dms\n", epoch, avgLoss, acc, elapsed);
        }
    }

    private static void connect(Node a, Node b) {
        b.addInput(a);
    }

    private static void logResult(String opt, int n, int trial, int epoch, long time, double mem, double acc,
            double loss) {
        try (PrintWriter writer = new PrintWriter(new FileWriter(RESULTS_FILE, true))) {
            writer.printf("%s,%d,%d,%d,%d,%.2f,%.4f,%.4f\n", opt, n, trial, epoch, time, mem, acc, loss);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static int argmax(Tensor t) {
        double max = Double.NEGATIVE_INFINITY;
        int idx = -1;
        for (int i = 0; i < t.data.length; i++) {
            if (t.data[i] > max) {
                max = t.data[i];
                idx = i;
            }
        }
        return idx;
    }
}
