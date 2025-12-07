package experiments;

import data.DataLoader;
import data.DataLoader.DataPoint;
import engine.*;
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
    private static final int BATCH_SIZE = 32; // Mini-batch size (simulated)
    // Note: Our layers currently support single-item forward/backward.
    // For true mini-batch, we'd need to accumulate gradients over BATCH_SIZE items
    // before update.

    private static final String TRAIN_FILE = "data/fashion-mnist_train.csv";
    private static final String TEST_FILE = "data/fashion-mnist_test.csv";
    private static final String RESULTS_FILE = "experiments/results.csv";

    public static void main(String[] args) throws IOException {
        // Initialize CSV
        try (PrintWriter writer = new PrintWriter(new FileWriter(RESULTS_FILE))) {
            writer.println("Optimizer,InputSize,Trial,Epoch,TimeMs,MemoryMB,Accuracy,Loss");
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
        List<DataPoint> testData = DataLoader.load(TEST_FILE, 1000); // Fixed test set size for speed

        // Build Model
        List<Layer> layers = new ArrayList<>();
        // Input: 1x28x28
        // Conv: 8 filters, 3x3 -> 8x26x26
        layers.add(new Conv2D(1, 8, 3));
        layers.add(new ReLU());
        // MaxPool: 2x2 -> 8x13x13
        layers.add(new MaxPool2D());
        // Dense: Input 8*13*13 = 1352 -> Output 10
        layers.add(new Dense(8 * 13 * 13, 10));
        layers.add(new Softmax());

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
                break; // Adam usually needs lower LR
            default:
                throw new IllegalArgumentException("Unknown optimizer");
        }

        CrossEntropyLoss lossFunc = new CrossEntropyLoss();
        CircularQueue lossQueue = new CircularQueue(100); // Moving average of last 100

        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            DataLoader.shuffle(trainData);
            double epochLoss = 0;
            int correct = 0;

            // Training Loop
            for (int i = 0; i < trainData.size(); i++) {
                DataPoint dp = trainData.get(i);

                // Forward
                Tensor out = dp.input;
                for (Layer l : layers) {
                    out = l.forward(out);
                }

                // Loss & Accuracy
                double l = lossFunc.forward(out, dp.label);
                epochLoss += l;
                lossQueue.add(l);

                int pred = argmax(out);
                if (pred == dp.labelIndex)
                    correct++;

                // Backward
                Tensor grad = lossFunc.backward(out, dp.label);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    grad = layers.get(j).backward(grad);
                }

                // Update (Simulated Batch: Update every step for now, or accumulate)
                // For pure SGD, we update every step.
                optimizer.update(layers);
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
            System.out.printf("Epoch %d: Loss=%.4f, Acc=%.4f, Time=%dms\n", epoch, avgLoss, acc, elapsed);
        }
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
