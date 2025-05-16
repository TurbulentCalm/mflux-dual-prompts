from mflux import Flux1Controlnet
from mflux.image_generator.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image based on a prompt and a controlnet reference image.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=True)
    parser.add_lora_arguments()
    parser.add_controlnet_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = ImageGenerator(model_class=Flux1Controlnet)
    generator.run(args)


if __name__ == "__main__":
    main()
