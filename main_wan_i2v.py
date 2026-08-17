import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

import torch
from PIL import Image
from myvllm.diffusion import DiffusionEngine


def main():
    config = {
        "model_name_or_path": "Wan-AI/Wan2.1-I2V-14B-720P",
        "world_size": 1,
        "runner_type": "diffusion",
        "text_encoder": "google/umt5-xxl",
        "diffusion": {
            "steps": 50,
            "shift": 5.0,
        },
    }

    engine = DiffusionEngine(config)

    image_path = "assets/cat.jpg"
    prompt = "the cat starts playing piano"

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}, using a dummy image")
        image = torch.randn(1, 3, 720, 1280)
    else:
        pil_image = Image.open(image_path).convert("RGB").resize((1280, 720))
        import torchvision.transforms as T
        transform = T.ToTensor()
        image = transform(pil_image).unsqueeze(0) * 2 - 1

    image = image.cuda()

    print(f"Generating video from image for: {prompt}")

    video = engine.i2v(
        image,
        prompt,
        num_frames=81,
        height=720,
        width=1280,
        seed=42,
    )

    if video is not None:
        print(f"Video shape: {video.shape}")
        import torchvision
        video_uint8 = ((video.squeeze(0).permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).byte()
        torchvision.io.write_video("output_i2v.mp4", video_uint8, fps=16)
        print("Saved to output_i2v.mp4")

    engine.exit()


if __name__ == "__main__":
    main()
