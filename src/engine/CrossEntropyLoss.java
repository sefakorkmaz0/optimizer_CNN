package engine;

public class CrossEntropyLoss {

    /**
     * Computes Cross Entropy Loss.
     * L = - sum(y_target * log(y_pred))
     */
    public double forward(Tensor predicted, Tensor target) {
        double loss = 0;
        double epsilon = 1e-15; // Prevent log(0)

        for (int i = 0; i < predicted.data.length; i++) {
            double p = Math.max(predicted.data[i], epsilon);
            loss -= target.data[i] * Math.log(p);
        }
        return loss;
    }

    /**
     * Computes gradient of Loss w.r.t Predicted output.
     * dL/dy_pred = - y_target / y_pred
     * 
     * Note: If combined with Softmax, the gradient is simpler (y_pred - y_target).
     * But since we have a separate Softmax layer, we return the raw gradient of CE.
     * The Softmax.backward() will handle the chain rule.
     */
    public Tensor backward(Tensor predicted, Tensor target) {
        Tensor grad = new Tensor(predicted.depth, predicted.height, predicted.width);
        double epsilon = 1e-15;

        for (int i = 0; i < predicted.data.length; i++) {
            double p = Math.max(predicted.data[i], epsilon);
            // dL/dp = -t/p
            grad.data[i] = -target.data[i] / p;
        }
        return grad;
    }
}
