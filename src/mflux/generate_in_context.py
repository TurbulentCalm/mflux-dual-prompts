from pathlib import Path
from mflux.community.in_context_lora.flux_in_context_lora import Flux1InContextLoRA
from mflux.image_generator.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


class InContextImageGenerator(ImageGenerator):
    def save_image(self, image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
        if getattr(args, "save_full_image", False):
            image.save(path=output_path.with_stem(output_path.stem + "_full"))


def main():
    parser = CommandLineParser(description="Generate an image using in-context LoRA with a reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = InContextImageGenerator(model_class=Flux1InContextLoRA)
    generator.run(args)


if __name__ == "__main__":
    main()
