import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D

# 1. Load latent vectors
def load_latents(latent_dir, max_frames=None):
    latent_files = sorted([
        f for f in os.listdir(latent_dir) if f.endswith(".pt")
    ])
    if max_frames:
        latent_files = latent_files[:max_frames]
    
    latents = []
    for f in latent_files:
        latent = torch.load(os.path.join(latent_dir, f))  # shape: (4, 64, 64)
        latents.append(latent.view(-1).numpy())  # flatten to (16384,)
    
    return np.stack(latents)

# 2. Dimensionality reduction
def reduce_dimensionality(latents, method="pca", n_components=2):
    if method == "pca":
        reducer = PCA(n_components=n_components)
    elif method == "tsne":
        reducer = TSNE(n_components=n_components, perplexity=30, init='pca', random_state=42)
    else:
        raise ValueError("method must be 'pca' or 'tsne'")
    
    return reducer.fit_transform(latents)

# 3. Visualization
def visualize_latents(reduced, method="pca"):
    if reduced.shape[1] == 2:
        plt.figure(figsize=(10, 6))
        plt.scatter(reduced[:, 0], reduced[:, 1], c=range(len(reduced)), cmap='viridis')
        plt.colorbar(label='Frame Index')
        plt.title(f'{method.upper()} 2D Visualization of Latent Space')
        plt.xlabel('Component 1')
        plt.ylabel('Component 2')
        plt.grid(True)
        plt.show()
    elif reduced.shape[1] == 3:
        fig = plt.figure(figsize=(10, 7))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(reduced[:, 0], reduced[:, 1], reduced[:, 2], c=range(len(reduced)), cmap='plasma')
        ax.set_title(f'{method.upper()} 3D Visualization of Latent Space')
        plt.show()
    else:
        raise ValueError("Reduced data must be 2D or 3D.")

# 4. Example usage
def run_visualization(video_path, max_frames=200):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    latent_dir = os.path.join("output", video_name)

    latents = load_latents(latent_dir, max_frames=max_frames)

    # PCA
    reduced_pca = reduce_dimensionality(latents, method="pca", n_components=2)
    visualize_latents(reduced_pca, method="pca")

    # t-SNE
    reduced_tsne = reduce_dimensionality(latents, method="tsne", n_components=2)
    visualize_latents(reduced_tsne, method="tsne")

# Run visualization for a specific video
run_visualization("rotation1.mp4")
