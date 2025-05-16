from mflux.flux_tools.depth.flux_depth import Flux1Depth
from mflux.image_generator.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using the depth tool.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_depth_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = ImageGenerator(model_class=Flux1Depth)
    generator.run(args)


if __name__ == "__main__":
    main()
