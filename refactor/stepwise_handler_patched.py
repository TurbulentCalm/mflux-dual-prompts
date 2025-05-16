from mflux.utils.prompt_utils import normalize_dual_prompts
from pathlib import Path
import logging

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback
from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil

log = logging.getLogger(__name__)

class StepwiseHandler(BeforeLoopCallback, InLoopCallback, InterruptCallback):
    def __init__(
        self,
        flux,
        output_dir: str,
        single_image: bool = False,
    ):
        self.flux = flux
        self.output_dir = Path(output_dir)
        self.step_wise_images = []
        self.single_image = single_image

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)

    def call_before_loop(
        self,
        seed: int,
        prompt: str = None,
        latents: mx.array = None,
        config: RuntimeConfig = None,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
        **kwargs,
    ) -> None:
        # Validate inputs
        if latents is None:
            log.error("call_before_loop received None latents")
            return
        if config is None:
            log.error("call_before_loop received None config")
            return
            
        clip_prompt = kwargs.get('clip_prompt', None)
        t5_prompt = kwargs.get('t5_prompt', None)
        dual_prompt = kwargs.get('dual_prompt', False)
        # Prefer dual prompt if enabled, else fallback to prompt
        used_prompt = prompt
        if dual_prompt:
            used_prompt = f"CLIP: {clip_prompt or ''} | T5: {t5_prompt or ''}"
            
        try:
            self._save_image(
                step=config.init_time_step,
                seed=seed,
                prompt=used_prompt,
                latents=latents,
                config=config,
                time_steps=None,
            )
        except Exception as e:
            log.error(f"Error in call_before_loop: {e}")

    def call_in_loop(
        self,
        t: int,
        seed: int,
        prompt: str = None,
        latents: mx.array = None,
        config: RuntimeConfig = None,
        time_steps: tqdm = None,
        **kwargs,
    ) -> None:
        # Validate inputs
        if latents is None:
            log.error("call_in_loop received None latents")
            return
        if config is None:
            log.error("call_in_loop received None config")
            return
            
        clip_prompt = kwargs.get('clip_prompt', None)
        t5_prompt = kwargs.get('t5_prompt', None)
        dual_prompt = kwargs.get('dual_prompt', False)
        used_prompt = prompt
        if dual_prompt:
            used_prompt = f"CLIP: {clip_prompt or ''} | T5: {t5_prompt or ''}"
            
        try:
            self._save_image(
                step=t + 1,
                seed=seed,
                prompt=used_prompt,
                latents=latents,
                config=config,
                time_steps=time_steps,
            )
        except Exception as e:
            log.error(f"Error in call_in_loop: {e}")

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str = None,
        latents: mx.array = None,
        config: RuntimeConfig = None,
        time_steps: tqdm = None,
        **kwargs,
    ) -> None:
        self._save_composite(seed=seed)

    def _save_image(
        self,
        step: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        log.info(f"Starting _save_image for step {step}")
        
        try:
            log.info(f"Unpacking latents of shape {latents.shape}...")
            unpack_latents = ArrayUtil.unpack_latents(latents=latents, height=config.height, width=config.width)
            if unpack_latents is None:
                log.error("Failed to unpack latents")
                return
            log.info(f"Unpacked latents shape: {unpack_latents.shape}")
                
            log.info("Decoding with VAE...")
            stepwise_decoded = self.flux.vae.decode(unpack_latents)
            if stepwise_decoded is None:
                log.error("VAE decode returned None")
                return
            log.info(f"Decoded shape: {stepwise_decoded.shape}")
                
            generation_time = time_steps.format_dict["elapsed"] if time_steps is not None else 0
            
            log.info("Converting to image...")
            stepwise_img = ImageUtil.to_image(
                decoded_latents=stepwise_decoded,
                config=config,
                seed=seed,
                prompt=prompt,
                quantization=self.flux.bits,
                lora_paths=self.flux.lora_paths,
                lora_scales=self.flux.lora_scales,
                generation_time=generation_time,
                dual_prompt=dual_prompt,
                clip_prompt=clip_prompt,
                t5_prompt=t5_prompt,
            )
            if stepwise_img is None:
                log.error("to_image returned None [DEBUG-TRACE] | stepwise_img: %s", stepwise_img)
                return
            if stepwise_img.image is None:
                log.error("Generated image has None image attribute [DEBUG-TRACE] | stepwise_img: %s", stepwise_img)
                return
            if not isinstance(stepwise_img.image, PIL.Image.Image):
                log.error(f"Generated image is not a PIL Image: {type(stepwise_img.image)} [DEBUG-TRACE] | stepwise_img.image: %s", stepwise_img.image)
                return
                
            log.info("Saving image... [DEBUG-TRACE] | stepwise_img: %s, stepwise_img.image: %s", stepwise_img, stepwise_img.image)
            try:
                # Ensure output directory exists
                self.output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save path
                save_path = (
                    self.output_dir / "current_step.png" if self.single_image 
                    else self.output_dir / f"seed_{seed}_step{step}of{config.num_inference_steps}.png"
                )
                
                # Verify image is valid before saving
                stepwise_img.image.verify()
                
                # Save image
                log.info(f"About to call stepwise_img.save [DEBUG-TRACE] | path: %s, stepwise_img: %s, stepwise_img.image: %s", save_path, stepwise_img, stepwise_img.image)
                stepwise_img.save(
                    path=save_path,
                    export_json_metadata=False,
                    overwrite=True if self.single_image else False,
                )
                
                # Verify file was created
                if not save_path.exists():
                    log.error(f"Image file was not created at {save_path}")
                    return
                    
                self.step_wise_images.append(stepwise_img)
                self._save_composite(seed=seed)
                log.info(f"Image saved successfully to {save_path}")
            except Exception as e:
                log.error(f"Failed to save image: {e} [DEBUG-TRACE]")
                import traceback
                log.error(f"Traceback: {traceback.format_exc()} [DEBUG-TRACE]")
        except Exception as e:
            log.error(f"Error in _save_image: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")

    def _save_composite(self, seed: int) -> None:
        if not self.step_wise_images:
            log.warning("No images to create composite from")
            return
            
        try:
            log.info("Creating composite image...")
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            if composite_img is None:
                log.error("Failed to create composite image")
                return
                
            # Save path
            save_path = (
                self.output_dir / "composite.png" if self.single_image
                else self.output_dir / f"seed_{seed}_composite.png"
            )
            
            # Ensure output directory exists
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Save composite
            composite_img.save(save_path)
            
            # Verify file was created
            if not save_path.exists():
                log.error(f"Composite image file was not created at {save_path}")
                return
                
            log.info(f"Composite image saved successfully to {save_path}")
        except Exception as e:
            log.error(f"Error saving composite image: {e}")
            import traceback
            log.error(f"Traceback: {traceback.format_exc()}")
