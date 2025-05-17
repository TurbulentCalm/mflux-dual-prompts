from pathlib import Path
from mflux import Config, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.callbacks.callbacks import Callbacks


class ImageGenerator:
    def __init__(
        self,
        model_class,
        model_config_args: dict = None
    ):
        self.model_class = model_class
        self.model_config_args = model_config_args or {}
        self.callback_registry = CallbackRegistry()
        self.memory_saver = None

    def save_image(
        self,
        image,
        seed,
        args
    ):
        # Override this method in generate_*.py for custom saving
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path, export_json_metadata=args.metadata)

    def optional_callbacks(
        self,
        flux,
        args
    ):
        # Override this method to add generator-specific callbacks
        pass

    def run(
        self,
        args,
        dual_prompts=False,
        clip_prompt=None,
        t5_prompt=None):
        model_config = ModelConfig.from_name(
            model_name=args.model,
            base_model=args.base_model
        )

        flux = self.model_class(
            model_config=model_config,
            quantize=args.quantize,
            local_path=args.path,
            lora_paths=args.lora_paths,
            lora_scales=args.lora_scales,
        )

        # Core callbacks that are always used
        if args.stepwise_image_output_dir:
            handler = StepwiseHandler(flux=flux, output_dir=args.stepwise_image_output_dir)
            self.callback_registry.register_before_loop(handler)
            self.callback_registry.register_in_loop(handler)
            self.callback_registry.register_interrupt(handler)

        if args.low_ram:
            self.memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
            self.callback_registry.register_before_loop(self.memory_saver)
            self.callback_registry.register_in_loop(self.memory_saver)
            self.callback_registry.register_after_loop(self.memory_saver)

        # Optional script-specific callbacks
        self.optional_callbacks(flux, args)

        try:
            for seed in args.seed:
                config = Config(
                    num_inference_steps=args.steps,
                    height=args.height,
                    width=args.width,
                    guidance=args.guidance,
                    image_path=args.image_path,
                    image_strength=args.image_strength,
                )
                generated_image, aux_outputs, runtime_config = flux.generate_image(
                    seed=seed,
                    prompt=args.prompt,
                    dual_prompts=dual_prompts,
                    clip_prompt=clip_prompt,
                    t5_prompt=t5_prompt,
                    config=config,
                )
                latents = aux_outputs.get('latents')
                aux_outputs = {k: v for k, v in aux_outputs.items() if k != 'latents'}
                if seed == args.seed[0]:
                    Callbacks.before_loop(
                        seed=seed,
                        prompt=args.prompt,
                        latents=latents,
                        config=runtime_config,
                        dual_prompts=dual_prompts,
                        clip_prompt=clip_prompt,
                        t5_prompt=t5_prompt,
                        **aux_outputs,
                    )
                    for cb in self.callback_registry.before_loop_callbacks():
                        cb.call_before_loop(
                            seed=seed,
                            prompt=args.prompt,
                            latents=latents,
                            config=runtime_config,
                            dual_prompts=dual_prompts,
                            clip_prompt=clip_prompt,
                            t5_prompt=t5_prompt,
                            **aux_outputs,
                        )
                self.save_image(generated_image, seed, args)
            # After the loop, call after_loop with the last values
            Callbacks.after_loop(
                seed=seed,
                prompt=args.prompt,
                latents=latents,
                config=runtime_config,
                dual_prompts=dual_prompts,
                clip_prompt=clip_prompt,
                t5_prompt=t5_prompt,
                **aux_outputs,
            )

        except StopImageGenerationException as stop_exc:
            print(stop_exc)

        finally:
            if self.memory_saver:
                print(self.memory_saver.memory_stats())
