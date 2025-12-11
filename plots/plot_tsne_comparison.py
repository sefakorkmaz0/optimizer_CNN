import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import os
import time

def plot_tsne(csv_path, ax, title):
    print(f"Loading {csv_path}...")
    if not os.path.exists(csv_path):
        print(f"File {csv_path} not found.")
        return

    df = pd.read_csv(csv_path)
    
    # Feature columns are d0..d9
    features = [f'd{i}' for i in range(10)]
    X = df[features].values
    y = df['label'].values
    
    print(f"Running t-SNE for {title}...")
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    X_embedded = tsne.fit_transform(X)
    
    df['tsne_1'] = X_embedded[:, 0]
    df['tsne_2'] = X_embedded[:, 1]
    
    sns.scatterplot(
        x='tsne_1', y='tsne_2',
        hue='label',
        palette=sns.color_palette("hls", 10),
        data=df,
        legend="full",
        alpha=0.7,
        ax=ax
    )
    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

def main():
    if not os.path.exists('plots'):
        os.makedirs('plots')

    fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    
    plot_tsne('experiments/embeddings_SGD.csv', axes[0], 'SGD Embeddings (t-SNE)')
    plot_tsne('experiments/embeddings_Adam.csv', axes[1], 'Adam Embeddings (t-SNE)')
    
    plt.tight_layout()
    output_file = 'plots/tsne_comparison.png'
    plt.savefig(output_file)
    print(f"Comparison plot saved to {output_file}")

if __name__ == "__main__":
    main()
