from pathlib import Path
from mflux import Config, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler


class ImageGenerator:
    def __init__(self, model_class, model_config_args: dict = None, extra_args_func=None):
        self.model_class = model_class
        self.model_config_args = model_config_args or {}
        self.extra_args_func = extra_args_func

        self.callback_registry = CallbackRegistry()
        self.memory_saver = None
        self.stepwise_handler = None

    def run(self, args):
        self.register_optional_callbacks = getattr(self, 'register_optional_callbacks', lambda flux, args: None)

        model_config = ModelConfig.from_name(
            model_name=args.model,
            base_model=args.base_model
        )

        flux = self.model_class(
            model_config=model_config,
            callbacks=self.callback_registry,
            quantize=args.quantize,
            local_path=args.path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )

        self.register_optional_callbacks(flux, args)

        try:
            for seed in args.seed:
                image = flux.generate_image(
                    seed=seed,
                    prompt=args.prompt,
                    dual_prompt=getattr(args, "dual_prompts", False),
                    clip_prompt=args.clip_prompt,
                    t5_prompt=args.t5_prompt,
                    config=Config(
                        num_inference_steps=args.steps,
                        height=args.height,
                        width=args.width,
                        guidance=args.guidance,
                        image_path=args.image_path,
                        image_strength=args.image_strength,
                    ),
                )
                self.save_image(image=image, seed=seed, args=args)
        except StopImageGenerationException as e:
            print(e)
        finally:
            if self.memory_saver:
                print(self.memory_saver.memory_stats())

    def save_image(self, image, seed, args):
        # Default saving logic — override in subclasses if needed
        image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)

    def register_optional_callbacks(self, flux, args):
        # This method can be overridden by subclasses to register tool-specific callbacks
        pass
