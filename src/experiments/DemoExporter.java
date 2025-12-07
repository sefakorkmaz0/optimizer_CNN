package experiments;

import data.DataLoader;
import data.DataLoader.DataPoint;
import engine.*;
import layers.*;
import utils.ModelExporter;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class DemoExporter {
    private static final int EPOCHS = 3;
    private static final String TRAIN_FILE = "data/fashion-mnist_train.csv";
    private static final String EXPORT_PATH = "demo/model_weights.json";

    public static void main(String[] args) throws IOException {
        // Create demo directory if not exists
        File demoDir = new File("demo");
        if (!demoDir.exists()) {
            demoDir.mkdirs();
        }

        // Load Data
        System.out.println("Loading data...");
        List<DataPoint> trainData = DataLoader.load(TRAIN_FILE, 60000);

        // Build Model
        List<Layer> layers = new ArrayList<>();
        layers.add(new Conv2D(1, 8, 3));
        layers.add(new ReLU());
        layers.add(new MaxPool2D());
        layers.add(new Dense(8 * 13 * 13, 10));
        layers.add(new Softmax());

        // Optimizer (Adam)
        Optimizer optimizer = new Adam(0.001);
        CrossEntropyLoss lossFunc = new CrossEntropyLoss();

        System.out.println("Training model for demo export...");
        long startTime = System.currentTimeMillis();

        for (int epoch = 1; epoch <= EPOCHS; epoch++) {
            DataLoader.shuffle(trainData);
            double epochLoss = 0;
            int correct = 0;

            for (int i = 0; i < trainData.size(); i++) {
                DataPoint dp = trainData.get(i);

                // Forward
                Tensor out = dp.input;
                for (Layer l : layers) {
                    out = l.forward(out);
                }

                // Loss & Accuracy
                epochLoss += lossFunc.forward(out, dp.label);
                if (argmax(out) == dp.labelIndex)
                    correct++;

                // Backward
                Tensor grad = lossFunc.backward(out, dp.label);
                for (int j = layers.size() - 1; j >= 0; j--) {
                    grad = layers.get(j).backward(grad);
                }

                // Update
                optimizer.update(layers);
            }

            long elapsed = System.currentTimeMillis() - startTime;
            double acc = (double) correct / trainData.size();
            double avgLoss = epochLoss / trainData.size();
            System.out.printf("Epoch %d: Loss=%.4f, Acc=%.4f, Time=%dms\n", epoch, avgLoss, acc, elapsed);
        }

        // Export
        System.out.println("Exporting weights...");
        ModelExporter.export(layers, EXPORT_PATH);
        System.out.println("Done!");
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
