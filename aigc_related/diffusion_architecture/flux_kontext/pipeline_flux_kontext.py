import inspect
from typing import Any, Callable
import numpy as np
import torch
from transformers import CLIPTextModel, CLIPTokenizer, T5EncoderModel, T5TokenizerFast
from diffusers.image_processor import PipelineImageInput, VaeImageProcessor
from diffusers.models.autoencoders import AutoencoderKL
from .transformer_flux import FluxTransformer2DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.flux.pipeline_output import FluxPipelineOutput

logger = logging.get_logger(__name__)

PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568), (688, 1504), (720, 1456), (752, 1392), (800, 1328),
    (832, 1248), (880, 1184), (944, 1104), (1024, 1024), (1104, 944),
    (1184, 880), (1248, 832), (1328, 800), (1392, 752), (1456, 720),
    (1504, 688), (1568, 672)
]

def calculate_shift(image_seq_len, base_seq_len: int=256, max_seq_len: int=4096, base_shift: float=0.5, max_shift: float=1.15):
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    mu = image_seq_len * m + b
    return mu

def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    """
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError('Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values')
        
    if timesteps is not None:
        accepts_timesteps = 'timesteps' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom timestep schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
        
    elif sigmas is not None:
        accept_sigmas = 'sigmas' in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom sigmas schedules. Please check whether you are using the correct scheduler.")
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
        
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        
    return (timesteps, num_inference_steps)

def retrieve_latents(encoder_output: torch.Tensor, generator: torch.Generator | None=None, sample_mode: str='sample'):
    if hasattr(encoder_output, 'latent_dist') and sample_mode == 'sample':
        return encoder_output.latent_dist.sample(generator)
    elif hasattr(encoder_output, 'latent_dist') and sample_mode == 'argmax':
        return encoder_output.latent_dist.mode()
    elif hasattr(encoder_output, 'latents'):
        return encoder_output.latents
    else:
        raise AttributeError('Could not access latents of provided encoder_output')

class FluxKontextPipeline(DiffusionPipeline):
    """
    The Flux Kontext pipeline for image-to-image and text-to-image generation.

    Reference: https://bfl.ai/announcements/flux-1-kontext-dev

    Args:
        transformer ([`FluxTransformer2DModel`]):
            Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded image latents.
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        text_encoder_2 ([`T5EncoderModel`]):
            [T5](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5EncoderModel), specifically
            the [google/t5-v1_1-xxl](https://huggingface.co/google/t5-v1_1-xxl) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/en/model_doc/clip#transformers.CLIPTokenizer).
        tokenizer_2 (`T5TokenizerFast`):
            Second Tokenizer of class
            [T5TokenizerFast](https://huggingface.co/docs/transformers/en/model_doc/t5#transformers.T5TokenizerFast).
    """
    model_cpu_offload_seq = 'text_encoder->text_encoder_2->image_encoder->transformer->vae'
    _optional_components = ['image_encoder', 'feature_extractor']
    _callback_tensor_inputs = ['latents', 'prompt_embeds']

    def __init__(
        self,
        scheduler: FlowMatchEulerDiscreteScheduler,
        vae: AutoencoderKL,
        text_encoder: CLIPTextModel,
        tokenizer: CLIPTokenizer,
        text_encoder_2: T5EncoderModel,
        tokenizer_2: T5TokenizerFast,
        transformer: FluxTransformer2DModel,
    ):
        super().__init__()
        
        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            text_encoder_2=text_encoder_2,
            tokenizer=tokenizer,
            tokenizer_2=tokenizer_2,
            transformer=transformer,
            scheduler=scheduler,
        )
        
        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, 'vae', None) else 8
        self.latent_channels = self.vae.config.latent_channels if getattr(self, 'vae', None) else 16
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor * 2)
        
        self.tokenizer_max_length = self.tokenizer.model_max_length if hasattr(self, 'tokenizer') and self.tokenizer is not None else 77
        self.default_sample_size = 128

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_images_per_prompt: int = 1,
        max_sequence_length: int = 512,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        text_inputs = self.tokenizer_2(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            return_length=False,
            return_overflowing_tokens=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer_2(prompt, padding='longest', return_tensors='pt').input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (not torch.equal(text_input_ids, untruncated_ids)):
            removed_text = self.tokenizer_2.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1:-1])
            logger.warning(f'The following part of your input was truncated because `max_sequence_length` is set to  {max_sequence_length} tokens: {removed_text}')
            
        prompt_embeds = self.text_encoder_2(text_input_ids.to(device), output_hidden_states=False)[0]
        dtype = self.text_encoder_2.dtype
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)
        
        (_, seq_len, _) = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
        
        return prompt_embeds

    def _get_clip_prompt_embeds(
        self,
        prompt: str | list[str],
        num_images_per_prompt: int = 1,
        device: torch.device | None = None,
    ):
        device = device or self._execution_device
        
        prompt = [prompt] if isinstance(prompt, str) else prompt
        batch_size = len(prompt)
        
        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=self.tokenizer_max_length,
            truncation=True,
            return_overflowing_tokens=False,
            return_length=False,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding='longest', return_tensors='pt').input_ids
        
        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and (not torch.equal(text_input_ids, untruncated_ids)):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer_max_length - 1:-1])
            logger.warning(f'The following part of your input was truncated because CLIP can only handle sequences up to {self.tokenizer_max_length} tokens: {removed_text}')
            
        prompt_embeds = self.text_encoder(text_input_ids.to(device), output_hidden_states=False)
        prompt_embeds = prompt_embeds.pooler_output
        prompt_embeds = prompt_embeds.to(dtype=self.text_encoder.dtype, device=device)
        prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt)
        prompt_embeds = prompt_embeds.view(batch_size * num_images_per_prompt, -1)
        
        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        prompt_2: str | list[str] | None = None,
        device: torch.device | None = None,
        num_images_per_prompt: int = 1,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        max_sequence_length: int = 512,
        lora_scale: float | None = None,
    ):
        """

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            prompt_2 (`str` or `list[str]`, *optional*):
                The prompt or prompts to be sent to the `tokenizer_2` and `text_encoder_2`. If not defined, `prompt` is
                used in all text-encoders
            device: (`torch.device`):
                torch device
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            pooled_prompt_embeds (`torch.FloatTensor`, *optional*):
                Pre-generated pooled text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting.
                If not provided, pooled text embeddings will be generated from `prompt` input argument.
            lora_scale (`float`, *optional*):
                A lora scale that will be applied to all LoRA layers of the text encoder if LoRA layers are loaded.
        """
        device = device or self._execution_device
        prompt = [prompt] if isinstance(prompt, str) else prompt
        
        if prompt_embeds is None:
            prompt_2 = prompt_2 or prompt
            prompt_2 = [prompt_2] if isinstance(prompt_2, str) else prompt_2
            
            pooled_prompt_embeds = self._get_clip_prompt_embeds(prompt=prompt, device=device, num_images_per_prompt=num_images_per_prompt)
            prompt_embeds = self._get_t5_prompt_embeds(prompt=prompt_2, num_images_per_prompt=num_images_per_prompt, max_sequence_length=max_sequence_length, device=device)
            
        dtype = self.text_encoder.dtype if self.text_encoder is not None else self.transformer.dtype
        text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(device=device, dtype=dtype)
        
        return (prompt_embeds, pooled_prompt_embeds, text_ids)

    @staticmethod
    def _prepare_latent_image_ids(batch_size, height, width, device, dtype):
        latent_image_ids = torch.zeros(height, width, 3)
        latent_image_ids[..., 1] = latent_image_ids[..., 1] + torch.arange(height)[:, None]
        latent_image_ids[..., 2] = latent_image_ids[..., 2] + torch.arange(width)[None, :]
        
        (latent_image_id_height, latent_image_id_width, latent_image_id_channels) = latent_image_ids.shape
        latent_image_ids = latent_image_ids.reshape(latent_image_id_height * latent_image_id_width, latent_image_id_channels)
        
        return latent_image_ids.to(device=device, dtype=dtype)

    @staticmethod
    def _pack_latents(latents, batch_size, num_channels_latents, height, width):
        latents = latents.view(batch_size, num_channels_latents, height // 2, 2, width // 2, 2)
        latents = latents.permute(0, 2, 4, 1, 3, 5)
        latents = latents.reshape(batch_size, height // 2 * (width // 2), num_channels_latents * 4)
        
        return latents

    @staticmethod
    def _unpack_latents(latents, height, width, vae_scale_factor):
        (batch_size, num_patches, channels) = latents.shape
        
        height = 2 * (int(height) // (vae_scale_factor * 2))
        width = 2 * (int(width) // (vae_scale_factor * 2))
        
        latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
        latents = latents.permute(0, 3, 1, 4, 2, 5)
        latents = latents.reshape(batch_size, channels // (2 * 2), height, width)
        
        return latents

    def _encode_vae_image(self, image: torch.Tensor, generator: torch.Generator):
        if isinstance(generator, list):
            image_latents = [
                retrieve_latents(
                    self.vae.encode(image[i : i + 1]),
                    generator=generator[i],
                    sample_mode="argmax",
                )
                for i in range(image.shape[0])
            ]
            image_latents = torch.cat(image_latents, dim=0)
        else:
            image_latents = retrieve_latents(self.vae.encode(image), generator=generator, sample_mode='argmax')
            
        image_latents = (image_latents - self.vae.config.shift_factor) * self.vae.config.scaling_factor
        return image_latents

    def prepare_latents(
        self,
        image: torch.Tensor | None,
        batch_size: int,
        num_channels_latents: int,
        height: int,
        width: int,
        dtype: torch.dtype,
        device: torch.device,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(f'You have passed a list of generators of length {len(generator)}, but requested an effective batch size of {batch_size}. Make sure the batch size matches the length of the generators.')
            
        height = 2 * (int(height) // (self.vae_scale_factor * 2))
        width = 2 * (int(width) // (self.vae_scale_factor * 2))
        shape = (batch_size, num_channels_latents, height, width)
        
        image_latents = image_ids = None
        if image is not None:
            image = image.to(device=device, dtype=dtype)
            
            if image.shape[1] != self.latent_channels:
                image_latents = self._encode_vae_image(image=image, generator=generator)
            else:
                image_latents = image
                
            if batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] == 0:
                additional_image_per_prompt = batch_size // image_latents.shape[0]
                image_latents = torch.cat([image_latents] * additional_image_per_prompt, dim=0)
            elif batch_size > image_latents.shape[0] and batch_size % image_latents.shape[0] != 0:
                raise ValueError(f'Cannot duplicate `image` of batch size {image_latents.shape[0]} to {batch_size} text prompts.')
            else:
                image_latents = torch.cat([image_latents], dim=0)
                
            (image_latent_height, image_latent_width) = image_latents.shape[2:]
            image_latents = self._pack_latents(image_latents, batch_size, num_channels_latents, image_latent_height, image_latent_width)
            
            image_ids = self._prepare_latent_image_ids(batch_size, image_latent_height // 2, image_latent_width // 2, device, dtype)
            image_ids[..., 0] = 1
            
        latent_ids = self._prepare_latent_image_ids(batch_size, height // 2, width // 2, device, dtype)
        
        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
            latents = self._pack_latents(latents, batch_size, num_channels_latents, height, width)
        else:
            latents = latents.to(device=device, dtype=dtype)
            
        return (latents, image_latents, latent_ids, image_ids)

    @torch.no_grad()
    def __call__(
        self,
        image: PipelineImageInput | None = None,
        prompt: str | list[str] = None,
        prompt_2: str | list[str] | None = None,
        negative_prompt: str | list[str] = None,
        negative_prompt_2: str | list[str] | None = None,
        true_cfg_scale: float = 1.0,
        height: int | None = None,
        width: int | None = None,
        num_inference_steps: int = 28,
        guidance_scale: float = 3.5,
        num_images_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        pooled_prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        negative_pooled_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str | None = "pil",
        return_dict: bool = True,
        max_sequence_length: int = 512,
    ):
        height = height or self.default_sample_size * self.vae_scale_factor
        width = width or self.default_sample_size * self.vae_scale_factor
        
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]
            
        device = self._execution_device
        
        (prompt_embeds, pooled_prompt_embeds, text_ids) = self.encode_prompt(
            prompt=prompt,
            prompt_2=prompt_2,
            prompt_embeds=prompt_embeds,
            pooled_prompt_embeds=pooled_prompt_embeds,
            device=device,
            num_images_per_prompt=num_images_per_prompt,
            max_sequence_length=max_sequence_length,
        )
        
        do_true_cfg = true_cfg_scale > 1.0
        if do_true_cfg:
            (negative_prompt_embeds, negative_pooled_prompt_embeds, negative_text_ids) = self.encode_prompt(
                prompt=negative_prompt,
                prompt_2=negative_prompt_2,
                prompt_embeds=negative_prompt_embeds,
                pooled_prompt_embeds=negative_pooled_prompt_embeds,
                device=device,
                num_images_per_prompt=num_images_per_prompt,
                max_sequence_length=max_sequence_length,
            )
        else:
            negative_prompt_embeds = None
            negative_pooled_prompt_embeds = None
            negative_text_ids = None
            
        image_latents = None
        if image is not None:
            image_latents = self._encode_vae_image(image=image, generator=generator)
            image_latents = self._pack_latents(
                image_latents,
                batch_size * num_images_per_prompt,
                image_latents.shape[1],
                image_latents.shape[2],
                image_latents.shape[3],
            )
            image_latents = image_latents.to(device=device, dtype=self.transformer.dtype)
            image_latent_ids = self._prepare_latent_image_ids(
                batch_size * num_images_per_prompt,
                height // 2,
                width // 2,
                device,
                self.transformer.dtype,
            )
            
        num_channels_latents = self.transformer.config.in_channels // 4
        (latents, latent_ids) = self.prepare_latents(
            None,
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )
        
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        image_seq_len = latents.shape[1]
        mu = calculate_shift(
            image_seq_len,
            self.scheduler.config.get("base_image_seq_len", 256),
            self.scheduler.config.get("max_image_seq_len", 4096),
            self.scheduler.config.get("base_shift", 0.5),
            self.scheduler.config.get("max_shift", 1.15),
        )
        (timesteps, num_inference_steps) = retrieve_timesteps(
            self.scheduler, num_inference_steps, device, None, sigmas, mu=mu
        )
        
        guidance = torch.full([1], guidance_scale, device=device, dtype=torch.float32)
        guidance = guidance.expand(latents.shape[0]) if guidance_scale > 1.0 else None
        
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            for (i, t) in enumerate(timesteps):
                latent_model_input = latents
                if image_latents is not None:
                    # Concat
                    latent_model_input = torch.cat((latents, image_latents), dim=1)
                timestep = t.expand(latents.shape[0]).to(latents.dtype)
                
                noise_pred = self.transformer(
                    hidden_states=latent_model_input,
                    timestep=timestep / 1000,
                    guidance=guidance,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    # Concat
                    img_ids=latent_ids if image is None else torch.cat((latent_ids, image_latent_ids), dim=1),
                    return_dict=False,
                )[0]
                noise_pred = noise_pred[:, :latents.size(1)]
                
                if do_true_cfg:
                    neg_noise_pred = self.transformer(
                        hidden_states=latent_model_input,
                        timestep=timestep / 1000,
                        guidance=guidance,
                        pooled_projections=negative_pooled_prompt_embeds,
                        encoder_hidden_states=negative_prompt_embeds,
                        txt_ids=negative_text_ids,
                        img_ids=latent_ids if image is None else torch.cat((latent_ids, image_latent_ids), dim=1),
                        return_dict=False,
                    )[0]
                    neg_noise_pred = neg_noise_pred[:, :latents.size(1)]
                    noise_pred = neg_noise_pred + true_cfg_scale * (noise_pred - neg_noise_pred)
                    
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]
                progress_bar.update()
                
        if output_type == 'latent':
            image = latents
        else:
            latents = self._unpack_latents(latents, height, width, self.vae_scale_factor)
            latents = latents / self.vae.config.scaling_factor + self.vae.config.shift_factor
            image = self.vae.decode(latents, return_dict=False)[0]
            image = self.image_processor.postprocess(image, output_type=output_type)
            
        if not return_dict:
            return (image,)
            
        return FluxPipelineOutput(images=image)