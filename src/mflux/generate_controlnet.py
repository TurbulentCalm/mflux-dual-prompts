from pathlib import Path

from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.models.flux_controlnet import Flux1Controlnet
from mflux.callbacks.instances.canny_image import CannyImageSaver
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using ControlNet guidance from a reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    clip_prompt, t5_prompt = normalize_dual_prompts(args)

    generator = ImageGenerator(model_class=Flux1Controlnet)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path, embed_metadata=getattr(args, 'embed_metadata', False), overwrite=getattr(args, 'overwrite_image', False))

    def register_optional_callbacks(flux, args):
        # If enabled, CannyImageSaver will save the canny image using the new callback system.
        if args.controlnet_save_canny:
            generator.callback_registry.register_before_loop(
                CannyImageSaver(path=args.output)
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
