import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from myvllm.diffusion import DiffusionEngine


def main():
    config = {
        "model_name_or_path": "Wan-AI/Wan2.1-T2V-14B",
        "world_size": 1,
        "runner_type": "diffusion",
        "text_encoder": "google/umt5-xxl",
        "diffusion": {
            "steps": 50,
            "shift": 5.0,
        },
    }

    engine = DiffusionEngine(config)

    prompt = "a cat playing piano on a beach at sunset"
    print(f"Generating video for: {prompt}")

    video = engine.t2v(
        prompt,
        num_frames=81,
        height=480,
        width=832,
        seed=42,
    )

    if video is not None:
        print(f"Video shape: {video.shape}")
        import torchvision
        video_uint8 = ((video.squeeze(0).permute(1, 2, 3, 0) + 1) * 127.5).clamp(0, 255).byte()
        torchvision.io.write_video("output_t2v.mp4", video_uint8, fps=16)
        print("Saved to output_t2v.mp4")

    engine.exit()


if __name__ == "__main__":
    main()
