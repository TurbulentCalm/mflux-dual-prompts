import mlx.core as mx
from mlx import nn
from tqdm import tqdm

from mflux.callbacks.callbacks import Callbacks
from mflux.config.config import Config
from mflux.config.model_config import ModelConfig
from mflux.config.runtime_config import RuntimeConfig
from mflux.controlnet.controlnet_util import ControlnetUtil
from mflux.controlnet.transformer_controlnet import TransformerControlnet
from mflux.error.exceptions import StopImageGenerationException
from mflux.flux.flux_initializer import FluxInitializer
from mflux.latent_creator.latent_creator import LatentCreator
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.prompt_encoder import PromptEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.models.transformer.transformer import Transformer
from mflux.models.vae.vae import VAE
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.generated_image import GeneratedImage
from mflux.post_processing.image_util import ImageUtil
from mflux.weights.model_saver import ModelSaver


class Flux1Controlnet(nn.Module):
    vae: VAE
    transformer: Transformer
    transformer_controlnet: TransformerControlnet
    t5_text_encoder: T5Encoder
    clip_text_encoder: CLIPEncoder

    def __init__(
        self,
        model_config: ModelConfig,
        quantize: int | None = None,
        local_path: str | None = None,
        lora_paths: list[str] | None = None,
        lora_scales: list[float] | None = None,
        controlnet_path: str | None = None,
    ):
        super().__init__()
        FluxInitializer.init_controlnet(
            flux_model=self,
            model_config=model_config,
            quantize=quantize,
            local_path=local_path,
            lora_paths=lora_paths,
            lora_scales=lora_scales,
        )

    def generate_image(
        self,
        seed: int,
        prompt: str = None,
        controlnet_image_path: str = None,
        config: Config = None,
        dual_prompt: bool = False,
        clip_prompt: str = None,
        t5_prompt: str = None,
    ) -> GeneratedImage:
        config = RuntimeConfig(config, self.model_config)
        time_steps = tqdm(range(config.num_inference_steps))
        controlnet_condition, canny_image = ControlnetUtil.encode_image(
            vae=self.vae,
            height=config.height,
            width=config.width,
            controlnet_image_path=controlnet_image_path,
        )
        latents = LatentCreator.create(
            seed=seed,
            height=config.height,
            width=config.width,
        )
        if dual_prompt:
            prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_dual_prompts(
                clip_prompt=clip_prompt,
                t5_prompt=t5_prompt,
                prompt_cache=self.prompt_cache,
                t5_tokenizer=self.t5_tokenizer,
                clip_tokenizer=self.clip_tokenizer,
                t5_text_encoder=self.t5_text_encoder,
                clip_text_encoder=self.clip_text_encoder,
            )
        else:
            prompt_embeds, pooled_prompt_embeds = PromptEncoder.encode_prompt(
                prompt=prompt,
                prompt_cache=self.prompt_cache,
                t5_tokenizer=self.t5_tokenizer,
                clip_tokenizer=self.clip_tokenizer,
                t5_text_encoder=self.t5_text_encoder,
                clip_text_encoder=self.clip_text_encoder,
            )
        Callbacks.before_loop(
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            canny_image=canny_image,
        )
        for t in time_steps:
            try:
                controlnet_block_samples, controlnet_single_block_samples = self.transformer_controlnet(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_condition=controlnet_condition,
                )
                noise = self.transformer(
                    t=t,
                    config=config,
                    hidden_states=latents,
                    prompt_embeds=prompt_embeds,
                    pooled_prompt_embeds=pooled_prompt_embeds,
                    controlnet_block_samples=controlnet_block_samples,
                    controlnet_single_block_samples=controlnet_single_block_samples,
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
            dual_prompt=dual_prompt,
            clip_prompt=clip_prompt,
            t5_prompt=t5_prompt,
            quantization=self.bits,
            lora_paths=self.lora_paths,
            lora_scales=self.lora_scales,
            controlnet_image_path=controlnet_image_path,
            generation_time=time_steps.format_dict["elapsed"],
        )

    def save_model(self, base_path: str) -> None:
        ModelSaver.save_model(self, self.bits, base_path)
        ModelSaver.save_weights(base_path, self.bits, self.transformer_controlnet, "transformer_controlnet")
