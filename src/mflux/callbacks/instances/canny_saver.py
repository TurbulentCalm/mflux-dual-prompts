import os
from pathlib import Path

import mlx.core as mx
import PIL.Image

from mflux import ImageUtil
from mflux.callbacks.callback import BeforeLoopCallback
from mflux.config.runtime_config import RuntimeConfig


class CannyImageSaver(BeforeLoopCallback):
    def __init__(self, path: str):
        self.path = Path(path)

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
        if canny_image is None:
            # *** Pre-loop or missing data: no canny image to save, skip saving to avoid errors
            return
        clip_prompt = kwargs.get('clip_prompt', None)
        t5_prompt = kwargs.get('t5_prompt', None)
        dual_prompts = kwargs.get('dual_prompts', False)
        used_prompt = prompt
        if dual_prompts:
            used_prompt = f"CLIP: {clip_prompt or ''} | T5: {t5_prompt or ''}"
        base, ext = os.path.splitext(self.path)
        ImageUtil.save_image(
            image=canny_image,
            path=f"{base}_controlnet_canny{ext}",
        )
