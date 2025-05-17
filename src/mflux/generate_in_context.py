
from pathlib import Path

from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.image_generator.image_generator import ImageGenerator
from mflux.community.in_context_lora.flux_in_context_lora import Flux1InContextLoRA
from mflux.community.in_context_lora.in_context_loras import get_lora_filename
from mflux.config.model_config import ModelConfig
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using in-context LoRA with a reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=True)
    parser.add_output_arguments()
    parser.add_argument(
        "--save-full-image",
        action="store_true",
        default=False,
        help="Additionally, save the full image including the reference (left side).",
    )
    parser.add_argument(
        "--lora-style",
        type=str,
        default=None,
        help="Name of the in-context LoRA style to apply."
    )
    args = parser.parse_args()

    clip_prompt, t5_prompt = normalize_dual_prompts(args)

    # Patch args to use fixed model config and LoRA filename list
    args.model = "flux_in_context"
    args.base_model = None
    args.path = None
    args.lora_paths = [get_lora_filename(args.lora_style)] if args.lora_style else None
    args.lora_scales = None

    generator = ImageGenerator(model_class=Flux1InContextLoRA)

    def save_image(image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        # Save right half (this is the generated image)
        image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
        if args.save_full_image:
            full_path = output_path.with_stem(output_path.stem + "_full")
            image.save(path=full_path)

    def optional_callbacks(flux, args):
        # No extra callbacks needed beyond the built-in memory and stepwise handlers
        pass

    generator.save_image = save_image
    generator.optional_callbacks = optional_callbacks

    generator.run(
        args,
        dual_prompts=False,
        clip_prompt=clip_prompt,
        t5_prompt=None,
    )

if __name__ == "__main__":
    main()
