
from pathlib import Path

from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.flux_tools.redux.flux_redux import Flux1Redux
from mflux.config.model_config import ModelConfig
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using the redux tool to fill masked regions.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=False)
    parser.add_redux_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    # Patch args for Redux fixed model
    args.model = "flux_redux"
    args.base_model = None

    clip_prompt, t5_prompt = normalize_dual_prompts(args)

    generator = ImageGenerator(model_class=Flux1Redux)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.save(path=output_path)

    def optional_callbacks(flux, args):
        # No additional callbacks required beyond standard memory and stepwise
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
