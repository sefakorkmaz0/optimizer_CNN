package utils;

import engine.Tensor;
import layers.Conv2D;
import layers.Dense;
import layers.Layer;
import java.io.FileWriter;
import java.io.IOException;
import java.io.PrintWriter;
import java.util.List;
import java.util.Locale;

public class ModelExporter {

    public static void export(List<Layer> layers, String filePath) {
        StringBuilder json = new StringBuilder();
        json.append("{\n");

        for (Layer layer : layers) {
            if (layer instanceof Conv2D) {
                Conv2D conv = (Conv2D) layer;
                List<Tensor> params = conv.getParameters();
                Tensor weights = params.get(0);
                Tensor biases = params.get(1);

                json.append("  \"conv2d\": {\n");
                json.append("    \"weights\": ").append(tensorToJson(weights)).append(",\n");
                json.append("    \"biases\": ").append(tensorToJson(biases)).append("\n");
                json.append("  },\n");
            } else if (layer instanceof Dense) {
                Dense dense = (Dense) layer;
                List<Tensor> params = dense.getParameters();
                Tensor weights = params.get(0);
                Tensor biases = params.get(1);

                json.append("  \"dense\": {\n");
                json.append("    \"weights\": ").append(tensorToJson(weights)).append(",\n");
                json.append("    \"biases\": ").append(tensorToJson(biases)).append("\n");
                json.append("  }\n"); // Last one, no comma
            }
        }

        json.append("}");

        try (PrintWriter out = new PrintWriter(new FileWriter(filePath))) {
            out.print(json.toString());
            System.out.println("Model weights exported to " + filePath);
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private static String tensorToJson(Tensor t) {
        StringBuilder sb = new StringBuilder();
        sb.append("[");
        for (int i = 0; i < t.data.length; i++) {
            sb.append(String.format(Locale.US, "%.6f", t.data[i]));
            if (i < t.data.length - 1) {
                sb.append(",");
            }
        }
        sb.append("]");
        return sb.toString();
    }
}
