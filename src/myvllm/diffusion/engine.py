import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from pathlib import Path
from transformers import AutoTokenizer, AutoModel

from myvllm.diffusion.dit import WanDiT
from myvllm.diffusion.vae import WanVAE
from myvllm.diffusion.scheduler import FlowScheduler
from myvllm.diffusion.pipeline import T2VPipeline, I2VPipeline
from myvllm.usp.group import init_usp_group, usp_rank, usp_world_size
from myvllm.utils import get_context


WAN_CONFIGS = {
    "Wan2.1-T2V-1.3B": dict(
        dim=1536, ffn_dim=8960, freq_dim=256, in_dim=16, out_dim=16,
        num_heads=12, num_layers=30, text_len=512,
    ),
    "Wan2.1-T2V-14B": dict(
        dim=5120, ffn_dim=13824, freq_dim=256, in_dim=16, out_dim=16,
        num_heads=40, num_layers=40, text_len=512,
    ),
    "Wan2.1-I2V-14B-480P": dict(
        dim=5120, ffn_dim=13824, freq_dim=256, in_dim=36, out_dim=16,
        num_heads=40, num_layers=40, text_len=512,
    ),
    "Wan2.1-I2V-14B-720P": dict(
        dim=5120, ffn_dim=13824, freq_dim=256, in_dim=36, out_dim=16,
        num_heads=40, num_layers=40, text_len=512,
    ),
}


def _worker_process(config, rank, port):
    import sys, os
    sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)
    sys.stderr = os.fdopen(sys.stderr.fileno(), 'w', buffering=1)
    torch.cuda.set_device(rank)
    init_usp_group(config["world_size"], rank, port=port)
    _run_loop(config, rank)


def _run_loop(config, rank):
    pass


class DiffusionEngine:
    def __init__(self, config):
        context = get_context()
        if context.runner_type not in ("generation", "pooling", "diffusion"):
            raise RuntimeError(
                "DiffusionEngine cannot coexist with another engine in the same process"
            )

        self.config = config
        world_size = config.get("world_size", 1)
        self.world_size = world_size
        usp_port = config.get("usp_port", 12346)

        model_name = Path(config["model_name_or_path"]).name
        for key in WAN_CONFIGS:
            if key in model_name:
                model_key = key
                break
        else:
            raise ValueError(f"Unsupported Wan model: {model_name}")

        self.model_config = WAN_CONFIGS[model_key]
        self.is_i2v = "I2V" in model_key

        if world_size > 1:
            ctx = mp.get_context("spawn")
            self.processes = []
            for i in range(1, world_size):
                process = ctx.Process(
                    target=_worker_process,
                    args=(config, i, usp_port),
                )
                self.processes.append(process)
                process.start()

        torch.cuda.set_device(0)
        init_usp_group(world_size, 0, port=usp_port)

        self.dit = WanDiT(**self.model_config).cuda(0)
        self.vae = WanVAE(config["model_name_or_path"]).to("cuda:0")

        text_encoder_name = config.get("text_encoder", "google/umt5-xxl")
        self.tokenizer = AutoTokenizer.from_pretrained(text_encoder_name)
        self.text_encoder = AutoModel.from_pretrained(text_encoder_name).cuda(0).eval()

        diffusion_cfg = config.get("diffusion", {})
        self.scheduler = FlowScheduler(
            steps=diffusion_cfg.get("steps", 50),
            shift=diffusion_cfg.get("shift", 5.0),
        )

        if self.is_i2v:
            self.pipeline = I2VPipeline(
                self.dit, self.vae, self.text_encoder, self.tokenizer,
                image_encoder=None, scheduler=self.scheduler, device="cuda:0",
            )
        else:
            self.pipeline = T2VPipeline(
                self.dit, self.vae, self.text_encoder, self.tokenizer,
                scheduler=self.scheduler, device="cuda:0",
            )

    def t2v(self, prompt, num_frames=81, height=480, width=832, seed=42):
        if self.is_i2v:
            raise RuntimeError("Use i2v() for I2V models")
        return self.pipeline(prompt, num_frames=num_frames, height=height, width=width, seed=seed)

    def i2v(self, image, prompt, num_frames=81, height=480, width=832, seed=42):
        if not self.is_i2v:
            raise RuntimeError("Use t2v() for T2V models")
        return self.pipeline(image, prompt, num_frames=num_frames, height=height, width=width, seed=seed)

    def exit(self):
        if self.world_size > 1:
            for process in self.processes:
                process.join()
