from pathlib import Path

from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_fill_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # Set default guidance for fill model
    if args.guidance is None:
        args.guidance = 30

    clip_prompt, t5_prompt = normalize_dual_prompts(args)

    generator = ImageGenerator(model_class=Flux1Fill)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path)

    def optional_callbacks(flux, args):
        # No additional callbacks needed for fill tasks at this time.
        # If mask overlays or intermediate maps are added to the model outputs, register a callback here.
        pass

    generator.save_image = save_image
    generator.optional_callbacks = optional_callbacks

    generator.run(
        args,
        dual_prompts=args.dual_prompts,
        clip_prompt=clip_prompt,
        t5_prompt=t5_prompt,
    )

if __name__ == "__main__":
    main()
