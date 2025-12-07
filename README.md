# Pure Java CNN Optimizer Benchmark

This project implements a Convolutional Neural Network (CNN) library from scratch in Java to benchmark different optimization algorithms (SGD, Momentum, Adam) on the Fashion-MNIST dataset.

## Project Structure
- `src/engine`: Core tensor operations and optimizer implementations.
- `src/layers`: Neural network layers (Conv2D, Dense, ReLU, MaxPool, Softmax).
- `src/data`: DataLoader for CSV datasets.
- `src/experiments`: Experiment runner and logging.
- `plots/`: Python scripts for visualization.

## Prerequisites
- Java JDK 11+
- Python 3 (for plotting) with `pandas`, `matplotlib`, `seaborn`
- Fashion-MNIST dataset (CSV format) in `data/` folder.

## Build & Run

### 1. Compile
```bash
javac -cp src src/engine/*.java src/layers/*.java src/utils/*.java src/data/*.java src/experiments/ExperimentRunner.java
```

### 2. Run Experiments
```bash
java -cp src experiments.ExperimentRunner
```
This will:
- Warm up the JVM.
- Run training for SGD, Momentum, and Adam.
- Test with Input Sizes N = 1000, 10000, 60000.
- Repeat each experiment 3 times.
- Save results to `experiments/results.csv`.

### 3. Generate Plots
```bash
python3 plots/plot_results.py
```
Plots will be saved in the `plots/` directory.

## Reproducibility
- **Seeds**: Random number generators are initialized with default seeds (or time-based) in the current implementation. For strict reproducibility, modify `Tensor.java` and `Layer` classes to accept a fixed seed.
- **Hardware**: Experiments were designed to run on standard consumer hardware (e.g., Intel Core i7, 16GB RAM).

## Algorithmic Analysis
The project compares:
- **Time Complexity**: $O(N)$ scaling.
- **Space Complexity**: SGD ($O(W)$) vs Adam ($O(3W)$).
- **Convergence**: Loss reduction over epochs.
