import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Create plots directory if not exists
if not os.path.exists('plots'):
    os.makedirs('plots')

# Load results
try:
    df = pd.read_csv('experiments/results.csv')
except FileNotFoundError:
    print("results.csv not found. Run the ExperimentRunner first.")
    exit()

# Set style
sns.set(style="whitegrid")

# 1. Runtime vs Input Size
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='InputSize', y='TimeMs', hue='Optimizer', marker='o', errorbar='sd')
plt.title('Training Runtime vs Input Size')
plt.ylabel('Time (ms)')
plt.xlabel('Input Size (N)')
plt.savefig('plots/runtime_vs_size.png')
plt.close()

# 2. Memory vs Input Size
plt.figure(figsize=(10, 6))
sns.lineplot(data=df, x='InputSize', y='MemoryMB', hue='Optimizer', marker='o', errorbar='sd')
plt.title('Peak Memory Usage vs Input Size')
plt.ylabel('Memory (MB)')
plt.xlabel('Input Size (N)')
plt.savefig('plots/memory_vs_size.png')
plt.close()

# 3. Loss vs Epoch (for largest N)
max_n = df['InputSize'].max()
df_max = df[df['InputSize'] == max_n]
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_max, x='Epoch', y='Loss', hue='Optimizer', marker='o')
plt.title(f'Loss Convergence (N={max_n})')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.savefig('plots/loss_convergence.png')
plt.close()

# 4. Accuracy vs Epoch (for largest N)
plt.figure(figsize=(10, 6))
sns.lineplot(data=df_max, x='Epoch', y='Accuracy', hue='Optimizer', marker='o')
plt.title(f'Accuracy Progression (N={max_n})')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.savefig('plots/accuracy_progression.png')
plt.close()

print("Plots generated in 'plots/' directory.")
