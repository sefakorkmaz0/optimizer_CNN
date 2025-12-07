package layers;

import engine.Tensor;
import java.util.ArrayList;
import java.util.List;

public class ReLU implements Layer {
    private Tensor lastInput;

    @Override
    public Tensor forward(Tensor input) {
        this.lastInput = input;
        Tensor output = new Tensor(input.depth, input.height, input.width);

        for (int i = 0; i < input.data.length; i++) {
            output.data[i] = Math.max(0, input.data[i]);
        }
        return output;
    }

    @Override
    public Tensor backward(Tensor gradOutput) {
        Tensor gradInput = new Tensor(lastInput.depth, lastInput.height, lastInput.width);

        for (int i = 0; i < gradOutput.data.length; i++) {
            // Derivative of ReLU is 1 if x > 0, else 0
            gradInput.data[i] = (lastInput.data[i] > 0) ? gradOutput.data[i] : 0;
        }
        return gradInput;
    }

    @Override
    public List<Tensor> getParameters() {
        return new ArrayList<>(); // No parameters
    }

    @Override
    public List<Tensor> getGradients() {
        return new ArrayList<>(); // No gradients
    }
}
