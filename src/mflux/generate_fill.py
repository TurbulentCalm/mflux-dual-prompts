from mflux.flux_tools.fill.flux_fill import Flux1Fill
from mflux.runner.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_fill_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = ImageGenerator(model_class=Flux1Fill)
    generator.run(args)


if __name__ == "__main__":
    main()
