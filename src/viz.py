import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def plot_clusters(X, labels, method="pca"):
    # Dimensionality reduction
    if method == "pca":
        reducer = PCA(n_components=2)
        X_reduced = reducer.fit_transform(X)
        title = "Cluster Visualization using PCA"
    elif method == "tsne":
        reducer = TSNE(n_components=2, random_state=42, perplexity=30, learning_rate=200)
        X_reduced = reducer.fit_transform(X)
        title = "Cluster Visualization using t-SNE"
    else:
        raise ValueError("method must be either 'pca' or 'tsne'")

    # ✅ Create a figure instead of using global plt
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(
        X_reduced[:, 0], X_reduced[:, 1],
        c=labels, cmap="tab10", alpha=0.7, edgecolor="k"
    )
    ax.set_title(title)
    ax.set_xlabel("Component 1")
    ax.set_ylabel("Component 2")
    ax.grid(True)
    fig.colorbar(scatter, ax=ax, label="Cluster")

    return fig  # ✅ return fig
