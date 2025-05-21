'''
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 1. Load latents
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

# 2. Compute metrics
def compute_l2_norm(latents):
    diffs = latents[1:] - latents[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    return norms

def compute_cos_similarity(latents):
    # Cosine similarity between consecutive frames
    sims = []
    for i in range(1, len(latents)):
        cos_sim = cosine_similarity([latents[i-1]], [latents[i]])[0, 0]
        sims.append(cos_sim)
    return np.array(sims)

def compute_mse(latents):
    diffs = latents[1:] - latents[:-1]
    mses = np.mean(diffs**2, axis=1)
    return mses

# 3. Visualization
def visualize_metrics(norms, cosine_sims, mses):
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    axes[0].scatter(np.arange(1, len(norms)+1), norms, c=norms, cmap='viridis')
    axes[0].set_title('L2 Norm (Magnitude of Latent Change)')
    axes[0].set_ylabel('L2 Norm')

    scatter = axes[1].scatter(np.arange(1, len(cosine_sims)+1), cosine_sims, c=cosine_sims, cmap='coolwarm')
    axes[1].set_title('Cosine Similarity Between Frames')
    axes[1].set_ylabel('Cosine Similarity')

    axes[2].scatter(np.arange(1, len(mses)+1), mses, c=mses, cmap='plasma')
    axes[2].set_title('Mean Squared Error Between Frames')
    axes[2].set_xlabel('Frame Index')
    axes[2].set_ylabel('MSE')

    plt.tight_layout()
    plt.show()

# 4. Run
def run_change_visualization(video_path, max_frames=200):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    latent_dir = os.path.join("output", video_name)

    latents = load_latents(latent_dir, max_frames=max_frames)
    l2 = compute_l2_norm(latents)
    cosine_sim = compute_cos_similarity(latents)
    mse = compute_mse(latents)
    
    visualize_metrics(l2, cosine_sim, mse)

# 실행
run_change_visualization("person.mp4")

'''
import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt

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

def extract_video_frames(video_path, max_frames=None, resize=(128, 128)):
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while cap.isOpened() and (max_frames is None or count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, resize)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame.astype(np.float32) / 255.0)  # normalize to [0, 1]
        count += 1
    cap.release()
    return np.stack(frames)  # shape: (N, H, W, C)

def compute_change_norms(latents):
    diffs = latents[1:] - latents[:-1]
    norms = np.linalg.norm(diffs, axis=1)
    return norms

def compute_image_mse(frames):
    diffs = frames[1:] - frames[:-1]
    mse = np.mean(diffs ** 2, axis=(1, 2, 3))  # mean over H, W, C
    return mse

def visualize_changes(change_norms, image_mse):
    plt.figure(figsize=(14, 5))

    plt.subplot(2, 1, 1)
    scatter1 = plt.scatter(np.arange(1, len(change_norms) + 1), change_norms, c=change_norms, cmap='plasma')
    plt.colorbar(scatter1, label='Latent Change Magnitude')
    plt.title('Latent Change Norms')
    plt.ylabel('Latent Norm')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    scatter2 = plt.scatter(np.arange(1, len(image_mse) + 1), image_mse, c=image_mse, cmap='viridis')
    plt.colorbar(scatter2, label='Image MSE')
    plt.title('Frame-to-Frame Image MSE')
    plt.xlabel('Frame Index')
    plt.ylabel('MSE')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

def run_change_visualization(video_path, max_frames=200):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    latent_dir = os.path.join("output", video_name)

    # Load latent vectors
    latents = load_latents(latent_dir, max_frames=max_frames)

    # Extract video frames
    frames = extract_video_frames(video_path, max_frames=max_frames)

    # Compute metrics
    change_norms = compute_change_norms(latents)
    image_mse = compute_image_mse(frames)

    # Visualize
    visualize_changes(change_norms, image_mse)

# Run
run_change_visualization("moon.mp4", max_frames=200)
