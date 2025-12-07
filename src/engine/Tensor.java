package engine;

import java.util.Arrays;
import java.util.Random;

/**
 * A 3D Tensor implementation using a flattened 1D array for cache efficiency.
 * Shape: [Depth, Height, Width]
 * 
 * Why 1D array?
 * Java's multi-dimensional arrays (double[][][]) are arrays of objects
 * (arrays),
 * which leads to memory fragmentation and poor cache locality.
 * A single 1D array ensures contiguous memory allocation, significantly
 * improving
 * performance for sequential access patterns common in convolution and matrix
 * operations.
 */
public class Tensor {
    public final double[] data;
    public final int depth;
    public final int height;
    public final int width;

    public Tensor(int depth, int height, int width) {
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.data = new double[depth * height * width];
    }

    public Tensor(int depth, int height, int width, double[] data) {
        if (data.length != depth * height * width) {
            throw new IllegalArgumentException("Data length does not match tensor shape");
        }
        this.depth = depth;
        this.height = height;
        this.width = width;
        this.data = data;
    }

    public int getIndex(int d, int h, int w) {
        return d * (height * width) + h * width + w;
    }

    public double get(int d, int h, int w) {
        return data[getIndex(d, h, w)];
    }

    public void set(int d, int h, int w, double value) {
        data[getIndex(d, h, w)] = value;
    }

    public void add(Tensor other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Tensor shape mismatch for addition");
        }
        for (int i = 0; i < data.length; i++) {
            this.data[i] += other.data[i];
        }
    }

    public void multiply(double scalar) {
        for (int i = 0; i < data.length; i++) {
            this.data[i] *= scalar;
        }
    }

    /**
     * Performs a dot product (element-wise multiplication and sum).
     * Used in Dense layers.
     */
    public double dot(Tensor other) {
        if (this.data.length != other.data.length) {
            throw new IllegalArgumentException("Tensor shape mismatch for dot product");
        }
        double sum = 0;
        for (int i = 0; i < data.length; i++) {
            sum += this.data[i] * other.data[i];
        }
        return sum;
    }

    public void fillRandom(double min, double max) {
        Random rand = new Random();
        for (int i = 0; i < data.length; i++) {
            data[i] = min + (max - min) * rand.nextDouble();
        }
    }

    public void fill(double value) {
        Arrays.fill(data, value);
    }

    public Tensor copy() {
        return new Tensor(depth, height, width, Arrays.copyOf(data, data.length));
    }

    @Override
    public String toString() {
        return "Tensor[" + depth + "x" + height + "x" + width + "]";
    }
}
