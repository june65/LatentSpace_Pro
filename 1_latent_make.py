import cv2
import torch
from diffusers.models import AutoencoderKL
from torchvision import transforms
from PIL import Image
import os
from tqdm import tqdm

# 1. Load Stable Diffusion's VAE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
vae.eval()

# 2. Define preprocessing transform
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),  # Stable Diffusion uses 512x512
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5]),  # [-1, 1] normalization
])

# 3. Function to encode image frame to latent vector
def encode_frame(frame_bgr):
    image = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    image_tensor = preprocess(image).unsqueeze(0).to(device)
    with torch.no_grad():
        latents = vae.encode(image_tensor).latent_dist.mean
    return latents.squeeze(0).cpu()#

# 4. Process video and encode each frame
def process_video(video_path, output_base_dir="output"):
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_idx = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        latent = encode_frame(frame)
        torch.save(latent, os.path.join(output_dir, f"frame_{frame_idx:05d}.pt"))
        frame_idx += 1

    cap.release()
    print(f"Saved {frame_idx} latent vectors to '{output_dir}'.")

# 5. Example usage
process_video("rotation.mp4")
