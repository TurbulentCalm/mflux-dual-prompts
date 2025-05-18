from pathlib import Path

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback
from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class StepwiseHandler(BeforeLoopCallback, InLoopCallback, InterruptCallback):
    def __init__(
        self,
        flux,
        output_dir: str,
        stepwise_single_image: bool = False,
    ):
        self.flux = flux
        self.output_dir = Path(output_dir)
        self.stepwise_images = []
        self.stepwise_single_image = stepwise_single_image

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        **kwargs,
    ) -> None:
        step = getattr(config, 'init_time_step', 0)  # *** Defensive: default to 0 if not present; pre-loop image is just a placeholder (blank or noise is fine)
        self._save_image(
            step=step,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=None,
        )

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
        **kwargs,
    ) -> None:
        self._save_image(
            step=t + 1,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=time_steps,
        )

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
        **kwargs,
    ) -> None:
        # *** Needs fixing no path argument
        self._save_composite(seed)

    def _save_image(
        self,
        step: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        if latents is None:
            # *** Pre-loop call: no latents to save, skip image generation (placeholder/blank image is fine)
            return
        unpack_latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
        stepwise_decoded = self.flux.vae.decode(unpack_latents)
        generation_time = time_steps.format_dict["elapsed"] if time_steps is not None else 0
        stepwise_img = ImageUtil.to_image(
            decoded_latents=stepwise_decoded,
            config=config,
            seed=seed,
            prompt=prompt,
            quantization=self.flux.bits,
            lora_paths=self.flux.lora_paths,
            lora_scales=self.flux.lora_scales,
            generation_time=generation_time,
        )
        if self.stepwise_single_image:
            stepwise_img.save(
                path=self.output_dir / f"current_step.png",
                export_json_metadata=False,
                overwrite=True,
            )
        else:
            stepwise_img.save(
                path=self.output_dir / f"seed_{seed}_step{step}of{config.num_inference_steps}.png",
                export_json_metadata=False,
                overwrite=False,
            )
        self.stepwise_images.append(stepwise_img)
        self._save_composite(seed)

    def _save_composite(self, seed: int) -> None:
        if self.stepwise_images:
            if self.stepwise_single_image:
                composite_img = ImageUtil.to_composite_image(self.stepwise_images)
                composite_img.save(self.output_dir / f"composite.png")
            else:
                composite_img = ImageUtil.to_composite_image(self.stepwise_images)
                composite_img.save(self.output_dir / f"seed_{seed}_composite.png")
