from mflux.flux_tools.redux.flux_redux import Flux1Redux
from mflux.image_generator.image_generator import ImageGenerator
from mflux.ui.cli.parsers import CommandLineParser


def main():
    parser = CommandLineParser(description="Generate an image using the fill tool to complete masked areas.")
    parser.add_general_arguments()
    parser.add_model_arguments(require_model_arg=False)
    parser.add_lora_arguments()
    parser.add_redux_arguments()
    parser.add_output_arguments()
    args = parser.parse_args()

    generator = ImageGenerator(model_class=Flux1Redux)
    generator.run(args)


if __name__ == "__main__":
    main()
