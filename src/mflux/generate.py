from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.flux.flux import Flux1
from mflux.ui.cli.parsers import CommandLineParser
from pathlib import Path


def main():
    parser = CommandLineParser(description="Generate an image based on prompts.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    clip_prompt, t5_prompt = normalize_dual_prompts(
        args.dual_prompts,
        args.prompt,
        args.clip_prompt,
        args.t5_prompt,
    )

    generator = ImageGenerator(model_class=Flux1)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path, export_json_metadata=args.metadata, embed_metadata=getattr(args, 'embed_metadata', False))

    def register_optional_callbacks(flux, args):
        # No additional callbacks needed for this script
        pass

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
