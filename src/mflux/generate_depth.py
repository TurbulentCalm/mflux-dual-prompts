from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.flux_tools.depth.flux_depth import Flux1Depth
from mflux.ui.cli.parsers import CommandLineParser
from mflux.callbacks.instances.depth_image import DepthImageSaver
from pathlib import Path


def main():
    parser = CommandLineParser(description="Generate an image using the depth tool.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_depth_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # Set default guidance for depth models
    if args.guidance is None:
        args.guidance = 10

    clip_prompt, t5_prompt = normalize_dual_prompts(args)

    generator = ImageGenerator(model_class=Flux1Depth)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path, embed_metadata=getattr(args, 'embed_metadata', False), overwrite=getattr(args, 'overwrite_image', False))

    def register_optional_callbacks(flux, args):
        # If enabled, DepthImageSaver will save the depth map using the new callback system.
        if args.save_depth_map:
            generator.callback_registry.register_before_loop(
                DepthImageSaver(path=args.output)
            )

    generator.save_image = save_image
    generator.register_optional_callbacks = register_optional_callbacks

    generator.run(
        args,
        dual_prompts=args.dual_prompts,
        clip_prompt=clip_prompt,
        t5_prompt=t5_prompt,
    )

if __name__ == "__main__":
    main()
