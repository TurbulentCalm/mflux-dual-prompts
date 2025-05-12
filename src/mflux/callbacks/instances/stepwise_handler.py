from pathlib import Path
import os
import glob
import time  # Add time module for timestamp

import mlx.core as mx
import PIL.Image
import tqdm

from mflux.callbacks.callback import BeforeLoopCallback, InLoopCallback, InterruptCallback, AfterLoopCallback
from mflux.config.runtime_config import RuntimeConfig
from mflux.post_processing.array_util import ArrayUtil
from mflux.post_processing.image_util import ImageUtil


class StepwiseHandler(BeforeLoopCallback, InLoopCallback, InterruptCallback, AfterLoopCallback):
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
        self.single_latest_image = None  # Store the latest image for single image mode
        self.composite_saved = False
        self.single_image_path = None
        
        # Generate a unique timestamp for this run
        self.timestamp = int(time.time())
        print(f"StepwiseHandler initialized with timestamp: {self.timestamp}")

        if self.output_dir:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            
        # For single image mode, clean up any existing files at initialization
        if self.single_image:
            self._cleanup_old_files()

    def _cleanup_old_files(self):
        """Remove any existing versioned files from previous runs"""
        try:
            # Clean up any versioned files from the single image mode
            pattern = str(self.output_dir / "current_progress*.png")
            for file_path in glob.glob(pattern):
                os.unlink(file_path)
                
            # Also clean any seed_X_latest* files that might exist
            for pattern in [
                str(self.output_dir / "seed_*_latest*.png"),
                str(self.output_dir / "seed_*_step*.png")
            ]:
                for file_path in glob.glob(pattern):
                    os.unlink(file_path)
        except Exception as e:
            print(f"Warning: Could not clean up old files: {e}")

    def call_before_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        canny_image: PIL.Image.Image | None = None,
        depth_image: PIL.Image.Image | None = None,
    ) -> None:
        self._save_image(
            step=config.init_time_step,
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
    ) -> None:
        self._save_image(
            step=t + 1,
            seed=seed,
            prompt=prompt,
            latents=latents,
            config=config,
            time_steps=time_steps,
        )

    def call_after_loop(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm = None,
    ) -> None:
        """Called after the generation loop completes successfully"""
        self._create_final_images(seed, prompt, latents, config, time_steps)

    def call_interrupt(
        self,
        t: int,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm,
    ) -> None:
        """Called if the generation is interrupted"""
        if self.single_image:
            self._create_final_images(seed, prompt, latents, config, time_steps)
        else:
            # Original behavior for multi-file mode
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
        
        if self.single_image:
            # In single image mode, just store the latest image and save it
            self.single_latest_image = stepwise_img
            
            # Use a unique timestamped filename that won't trigger versioning
            self.single_image_path = self.output_dir / f"progress_{self.timestamp}_{step}.png"
            print(f"Saving step {step} to {self.single_image_path}")
            
            # Only maintain a single file for each step
            stepwise_img.save(
                path=self.single_image_path,
                export_json_metadata=False,
            )
        else:
            # Default behavior: save a separate file for each step
            stepwise_img.save(
                path=self.output_dir / f"seed_{seed}_step{step}of{config.num_inference_steps}.png",
                export_json_metadata=False,
            )
            # Only append to step_wise_images in multi-file mode
            self.step_wise_images.append(stepwise_img)
            self._save_composite(seed=seed)

    def _save_composite(self, seed: int) -> None:
        if self.step_wise_images:
            composite_img = ImageUtil.to_composite_image(self.step_wise_images)
            composite_img.save(self.output_dir / f"seed_{seed}_composite.png")

    def _create_final_images(
        self,
        seed: int,
        prompt: str,
        latents: mx.array,
        config: RuntimeConfig,
        time_steps: tqdm = None,
    ) -> None:
        """Create the final and composite images, and clean up temporary files"""
        if self.single_image and not self.composite_saved and self.single_latest_image is not None:
            # Save the final image with the proper seed name
            final_path = self.output_dir / f"seed_{seed}_final.png"
            print(f"Saving final image to {final_path}")
            self.single_latest_image.save(
                path=final_path,
                export_json_metadata=False,
            )
            
            # Create a new composite image from the single latest image
            composite_path = self.output_dir / f"seed_{seed}_composite.png"
            print(f"Saving composite image to {composite_path}")
            composite_img = ImageUtil.to_composite_image([self.single_latest_image])
            composite_img.save(composite_path)
            self.composite_saved = True
            
            # Remove all progress files since we're done
            print(f"Cleaning up progress files...")
            try:
                # Clean up all progress files with this timestamp
                pattern = str(self.output_dir / f"progress_{self.timestamp}_*.png")
                for file_path in glob.glob(pattern):
                    print(f"Removing {file_path}")
                    os.unlink(file_path)
            except Exception as e:
                print(f"Warning: Error cleaning up progress files: {e}")
