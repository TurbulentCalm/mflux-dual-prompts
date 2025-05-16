from mflux import Flux1
from mflux.image_generator.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image based on a prompt.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_image_generator_arguments(supports_metadata_config=True)
    parser.add_image_to_image_arguments(required=False)
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = ImageGenerator(model_class=Flux1)
    generator.run(args)


if __name__ == "__main__":
    main()
