from mflux import Config, Flux1Controlnet, ModelConfig, StopImageGenerationException
from mflux.callbacks.callback_registry import CallbackRegistry
from mflux.callbacks.instances.canny_saver import CannyImageSaver
from mflux.callbacks.instances.memory_saver import MemorySaver
from mflux.callbacks.instances.stepwise_handler import StepwiseHandler
from mflux.ui.cli.parsers import CommandLineParser


def main():
    # 0. Parse command line arguments
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")  # fmt: off
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_controlnet_arguments()
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_output_arguments()
    args = parser.parse_args()

    # 1. Load the model
    flux = Flux1Controlnet(
        model_config=ModelConfig.from_name(model_name=args.model, base_model=args.base_model),
        quantize=args.quantize,
        local_path=args.path,
        lora_paths=args.lora_paths,
        lora_scales=args.lora_scales,
    )

    # 2. Register the optional callbacks
    if args.controlnet_save_canny:
        CallbackRegistry.register_before_loop(CannyImageSaver(path=args.output))
    if args.stepwise_image_output_dir:
        handler = StepwiseHandler(
            flux=flux, 
            output_dir=args.stepwise_image_output_dir,
            single_image=getattr(args, 'stepwise_single_image', False)
        )
        CallbackRegistry.register_before_loop(handler)
        CallbackRegistry.register_in_loop(handler)
        CallbackRegistry.register_interrupt(handler)
        CallbackRegistry.register_after_loop(handler)

    memory_saver = None
    if args.low_ram:
        memory_saver = MemorySaver(flux=flux, keep_transformer=len(args.seed) > 1)
        CallbackRegistry.register_before_loop(memory_saver)
        CallbackRegistry.register_in_loop(memory_saver)
        CallbackRegistry.register_after_loop(memory_saver)

    try:
        for seed in args.seed:
            dual_prompts = getattr(args, 'dual_prompts', False)
            clip_l_prompt = getattr(args, 'clip_l_prompt', "")
            t5_prompt = getattr(args, 't5_prompt', "")
            if dual_prompts:
                image = flux.generate_image(
                    seed=seed,
                    clip_l_prompt=clip_l_prompt,
                    t5_prompt=t5_prompt,
                    dual_prompts=True,
                    controlnet_image_path=args.controlnet_image_path,
                    config=Config(
                        num_inference_steps=args.steps,
                        height=args.height,
                        width=args.width,
                        guidance=args.guidance,
                        controlnet_strength=args.controlnet_strength,
                    ),
                )
            else:
                image = flux.generate_image(
                    seed=seed,
                    prompt=args.prompt,
                    controlnet_image_path=args.controlnet_image_path,
                    config=Config(
                        num_inference_steps=args.steps,
                        height=args.height,
                        width=args.width,
                        guidance=args.guidance,
                        controlnet_strength=args.controlnet_strength,
                    ),
                )

            # 4. Save the image
            image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
    except StopImageGenerationException as stop_exc:
        print(stop_exc)
    finally:
        if memory_saver:
            print(memory_saver.memory_stats())


if __name__ == "__main__":
    main()
