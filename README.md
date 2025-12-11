# CNN Optimizer Benchmark - From Scratch Implementation

A pure Java implementation of a Convolutional Neural Network (CNN) library with a DAG-based computation graph and multiple optimization algorithms. This project benchmarks SGD, Momentum, and Adam optimizers on the Fashion-MNIST dataset.

## 🎯 Project Goals

1. **Algorithm Implementation**: Build backpropagation and optimization algorithms from scratch
2. **Data Structures**: Implement a DAG-based computation graph with topological sorting
3. **Empirical Analysis**: Compare optimizer performance across different dataset sizes
4. **Visualization**: Demonstrate learned feature representations via t-SNE

## 📁 Project Structure

```
optimizer_CNN/
├── src/
│   ├── engine/          # Core: Tensor, Optimizers (SGD, Adam), Loss
│   │   └── graph/       # ComputationGraph, Node (DAG implementation)
│   ├── layers/          # Conv2D, Dense, MaxPool2D, ReLU, Softmax
│   ├── data/            # DataLoader for CSV datasets
│   ├── experiments/     # ExperimentRunner, EmbeddingExtractor
│   ├── tests/           # Unit tests for layers and graph
│   └── utils/           # CircularQueue, ModelExporter
├── data/                # Fashion-MNIST CSV files (not in git)
├── experiments/         # Generated results (CSV)
├── plots/               # Visualization scripts and outputs
├── AI_USAGE.md          # AI tool usage declaration
└── README.md            # This file
```

## 🛠️ Prerequisites

- **Java**: JDK 11 or higher
- **Python 3**: For plotting (with `pandas`, `matplotlib`, `seaborn`, `scikit-learn`)
- **Dataset**: Fashion-MNIST in CSV format in `data/` folder
  - Download from: https://www.kaggle.com/datasets/zalando-research/fashionmnist

## 🚀 Build & Run

### 1. Compile

```bash
# Create bin directory
mkdir -p bin

# Compile all Java files
javac -cp src -d bin src/engine/*.java src/engine/graph/*.java src/layers/*.java src/utils/*.java src/data/*.java src/experiments/*.java src/tests/*.java
```

### 2. Run Tests (Verify Installation)

```bash
java -cp bin tests.LayerTest
java -cp bin tests.GraphTest
java -cp bin tests.TensorTest
```

### 3. Run Benchmark Experiments

```bash
java -cp bin experiments.ExperimentRunner
```

This will:
- Warm up the JVM
- Train with SGD, Momentum, and Adam optimizers
- Test with input sizes: 1,000 / 10,000 / 60,000
- Run 3 trials per configuration
- Save results to `experiments/results.csv`

### 4. Generate Embedding Comparison

```bash
java -cp bin experiments.EmbeddingExtractor
```

### 5. Generate Plots

```bash
# Benchmark plots
python3 plots/plot_results.py

# t-SNE comparison
python3 plots/plot_tsne_comparison.py
```

## 📊 Results

### Benchmark Metrics

| Optimizer | Final Accuracy (60k) | Convergence Speed | Memory Usage |
|-----------|---------------------|-------------------|--------------|
| SGD       | ~89%                | Slow              | O(W)         |
| Momentum  | ~75-85%*            | Medium            | O(2W)        |
| Adam      | ~90.6%              | Fast              | O(3W)        |

*Momentum showed instability with default hyperparameters on this architecture.

### Plots Generated
- `plots/runtime_vs_size.png` - Training time vs dataset size
- `plots/memory_vs_size.png` - Memory usage comparison
- `plots/loss_convergence.png` - Loss over epochs
- `plots/accuracy_progression.png` - Accuracy over epochs
- `plots/tsne_comparison.png` - Feature space visualization (SGD vs Adam)

## 🔬 Reproducibility

### Hardware
- **CPU**: [Your CPU Here, e.g., Intel Core i7-10700]
- **RAM**: [Your RAM Here, e.g., 16GB DDR4]
- **OS**: [Your OS Here, e.g., Ubuntu 22.04]

### Random Seeds
The current implementation uses time-based random initialization. For strict reproducibility, modify:
- `Tensor.java:fillRandom()` - Add seed parameter
- `Conv2D.java`, `Dense.java` - Use seeded Random instance

### Trials
Each experiment is repeated 3 times with results averaged (variance shown in plots).

## 📐 Algorithmic Analysis

### Time Complexity

| Component | Complexity |
|-----------|------------|
| Forward Pass (Conv2D) | O(F × D × K² × H × W) |
| Forward Pass (Dense) | O(I × O) |
| Backward Pass | Same as forward |
| Topological Sort | O(V + E) |
| SGD Update | O(W) |
| Adam Update | O(W) |

Where: F=filters, D=depth, K=kernel, H×W=output size, I=input, O=output, V=nodes, E=edges, W=parameters

### Space Complexity

| Optimizer | Additional Memory |
|-----------|------------------|
| SGD | O(1) |
| Momentum | O(W) - velocity |
| Adam | O(2W) - m and v vectors |

## 🎥 Demo Video

[Link to demo video - 2-5 minutes]

## 📚 References

- [Fashion-MNIST Dataset](https://github.com/zalandoresearch/fashion-mnist)
- [Adam Optimizer Paper](https://arxiv.org/abs/1412.6980)
- [Backpropagation - Wikipedia](https://en.wikipedia.org/wiki/Backpropagation)

## 👥 Team

- [Your Name Here]

## 📄 License

This project is for educational purposes as part of the Algorithms Course (Fall 2025).
