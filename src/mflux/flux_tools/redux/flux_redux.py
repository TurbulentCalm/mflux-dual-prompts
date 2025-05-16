from pathlib import Path

import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.flux_tools.redux.redux_util import ReduxUtil
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.redux_encoder.redux_encoder import ReduxEncoder
from mflux.models.siglip_vision_transformer.siglip_vision_transformer import SiglipVisionTransformer
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class Flux1Redux(nn.Module):
    vae: VAE
    image_encoder: SiglipVisionTransformer
    image_embedder: ReduxEncoder
    transformer: Transformer
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
    ):
        super().__init__()
        FluxInitializer.init_redux(
            flux_model=self,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str = None,
        config: Config = None,
        dual_prompt: bool = False,
        clip_prompt: str = None,
        t5_prompt: str = None,
    ) -> GeneratedImage:
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.init_time_step, config.num_inference_steps))
        latents = LatentCreator.create(
            seed=seed,
            height=config.height,
            width=config.width,
        )
        prompt_embeds, pooled_prompt_embeds = self._get_prompt_embeddings(
            prompt=prompt,
            dual_prompt=dual_prompt,
            clip_prompt=clip_prompt,
            t5_prompt=t5_prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
            image_paths=config.redux_image_paths,
            image_encoder=self.image_encoder,
            image_embedder=self.image_embedder,
        )
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )
        for t in time_steps:
            try:
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                )
                dt = config.sigmas[t + 1] - config.sigmas[t]
                latents += noise * dt
                Callbacks.in_loop(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )
                mx.eval(latents)
            except KeyboardInterrupt:
                Callbacks.interruption(
                    t=t,
                    seed=seed,
                    prompt=prompt,
                    latents=latents,
                    config=config,
                    time_steps=time_steps,
                )
                raise StopImageGenerationException(f"Stopping image generation at step {t + 1}/{len(time_steps)}")
        Callbacks.after_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
        )
        latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(
            decoded_latents=decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            redux_image_paths=config.redux_image_paths,
            image_strength=config.image_strength,
            generation_time=time_steps.format_dict["elapsed"],
            dual_prompt=dual_prompt,
            clip_prompt=clip_prompt,
            t5_prompt=t5_prompt,
        )

    @staticmethod
    def _get_prompt_embeddings(
        prompt: str = None,
        dual_prompt: bool = False,
        clip_prompt: str = None,
        t5_prompt: str = None,
        prompt_cache: dict = None,
        t5_tokenizer: T5Tokenizer = None,
        clip_tokenizer: TokenizerCLIP = None,
        t5_text_encoder: T5Encoder = None,
        clip_text_encoder: CLIPEncoder = None,
        image_paths: list = None,
        image_encoder: SiglipVisionTransformer = None,
        image_embedder: ReduxEncoder = None,
    ) -> tuple[mx.array, mx.array]:
        prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompts(
            clip_prompt=clip_prompt,
            t5_prompt=t5_prompt,
            prompt_cache=self.prompt_cache,
            t5_tokenizer=self.t5_tokenizer,
            clip_tokenizer=self.clip_tokenizer,
            t5_text_encoder=self.t5_text_encoder,
            clip_text_encoder=self.clip_text_encoder,
        )
        image_embeds = ReduxUtil.embed_images(
            image_paths=image_paths,
            image_encoder=image_encoder,
            image_embedder=image_embedder,
        )
        prompt_embeds = mx.concatenate([prompt_embeds_txt] + image_embeds, axis=1)
        return prompt_embeds, pooled_prompt_embeds
