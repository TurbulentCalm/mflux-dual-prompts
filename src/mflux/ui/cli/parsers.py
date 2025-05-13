import argparse
import json
import random
import time
import typing as t
import sys
import re
from pathlib import Path

from mflux.community.in_context_lora.in_context_loras import LORA_NAME_MAP, LORA_REPO_ID
from mflux.ui import (
    box_values,
    defaults as ui_defaults,
)


def preprocess_args(args):
    """Process command line arguments, handling backslash continuations that join values with flags."""
    processed_args = []
    i = 0
    while i < len(args):
        arg = args[i]
        
        # Check if this argument might contain a concatenated flag due to backslash
        backslash_value_match = re.match(r'^(.*?)\s+(--\w[\w-]*)(.*)$', arg)
        
        if backslash_value_match:
            # Handle separated values and flags
            value, flag, rest = backslash_value_match.groups()
            
            # Add the value
            processed_args.append(value)
            
            # Add the flag (and its value if it has one)
            if rest:
                processed_args.append(f"{flag}{rest}")
            else:
                processed_args.append(flag)
                # Check if the next arg is a flag value and not a flag
                if i + 1 < len(args) and not args[i+1].startswith('--'):
                    processed_args.append(args[i+1])
                    i += 1
        else:
            # Just a normal argument
            processed_args.append(arg)
        
        i += 1
    
    return processed_args


def safe_float(value):
    """Convert a string to float, handling whitespace and other issues.
    Returns None for empty strings after stripping whitespace."""
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
    try:
        return float(value)
    except (ValueError, TypeError):
        # If we can't convert, return the original value and let argparse handle the error
        return value


class LoraScalesAction(argparse.Action):
    """Custom action for lora-scales to filter out None/empty values and split space-separated values."""
    def __call__(self, parser, namespace, values, option_string=None):
        # Filter out None values (which come from empty strings)
        if values is None:
            setattr(namespace, self.dest, None)
            return
        
        # Process values, handling both individual floats and space-separated values
        processed_values = []
        for val in values:
            if val is None:
                continue
                
            # Check if this is a space-separated list of values (like "1.2 0.7")
            if isinstance(val, str) and ' ' in val.strip():
                # Split and convert each part to float
                try:
                    parts = [float(part.strip()) for part in val.split() if part.strip()]
                    processed_values.extend(parts)
                except ValueError:
                    # If conversion fails, just add the original value
                    processed_values.append(val)
            else:
                processed_values.append(val)
        
        if not processed_values:
            setattr(namespace, self.dest, None)  # If all values were filtered out, use None
        else:
            setattr(namespace, self.dest, processed_values)


class ModelSpecAction(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        if values in ui_defaults.MODEL_CHOICES:
            setattr(namespace, self.dest, values)
            return

        if values is None or values.count("/") != 1:
            raise argparse.ArgumentError(
                self,
                (f'Value must be either {" ".join(ui_defaults.MODEL_CHOICES)} or in format "org/model". Got: {values}'),
            )

        # If we got here, values contains exactly one slash
        setattr(namespace, self.dest, values)


# fmt: off
class CommandLineParser(argparse.ArgumentParser):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.supports_metadata_config = False
        self.supports_image_generation = False
        self.supports_controlnet = False
        self.supports_image_to_image = False
        self.supports_image_outpaint = False
        self.supports_lora = False
        self.require_model_arg = True

    def add_general_arguments(self) -> None:
        self.add_argument("--low-ram", action="store_true", help="Enable low-RAM mode to reduce memory usage (may impact performance).")
        self.add_argument("--debug", action="store_true", help="Print debug information about command arguments and execution.")

    def add_model_arguments(self, path_type: t.Literal["load", "save"] = "load", require_model_arg: bool = True) -> None:
        self.require_model_arg = require_model_arg
        self.add_argument("--model", "-m", type=str, required=require_model_arg, action=ModelSpecAction, help=f"The model to use ({' or '.join(ui_defaults.MODEL_CHOICES)} or a compatible huggingface repo_id org/model).")
        if path_type == "load":
            self.add_argument("--path", type=str, default=None, help="Local path for loading a model from disk")
        else:
            self.add_argument("--path", type=str, required=True, help="Local path for saving a model to disk.")
        self.add_argument("--base-model", type=str, required=False, choices=ui_defaults.MODEL_CHOICES, help="When using a third-party huggingface model, explicitly specify whether the base model is dev or schnell")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_lora_arguments(self) -> None:
        self.supports_lora = True
        lora_group = self.add_argument_group("LoRA configuration")
        lora_group.add_argument("--lora-style", type=str, choices=sorted(LORA_NAME_MAP.keys()), help="Style of the LoRA to use (e.g., 'storyboard' for film storyboard style)")
        self.add_argument("--lora-paths", type=str, nargs="*", default=None, help="Local safetensors for applying LORA from disk")
        self.add_argument("--lora-scales", type=float, nargs="*", default=None, help="Scaling factor to adjust the impact of LoRA weights on the model. A value of 1.0 applies the LoRA weights as they are.")
        lora_group.add_argument("--lora-name", type=str, help="Name of the LoRA to download from Hugging Face")
        lora_group.add_argument("--lora-repo-id", type=str, default=LORA_REPO_ID, help=f"Hugging Face repository ID for LoRAs (default: {LORA_REPO_ID})")

    def _add_image_generator_common_arguments(self) -> None:
        self.supports_image_generation = True
        self.add_argument("--height", type=int, default=ui_defaults.HEIGHT, help=f"Image height (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--width", type=int, default=ui_defaults.WIDTH, help=f"Image width (Default is {ui_defaults.HEIGHT})")
        self.add_argument("--steps", type=int, default=None, help="Inference Steps")
        self.add_argument("--guidance", type=float, default=ui_defaults.GUIDANCE_SCALE, help=f"Guidance Scale (Default is {ui_defaults.GUIDANCE_SCALE})")

    def add_image_generator_arguments(self, supports_metadata_config=False) -> None:
        prompt_group = self.add_argument_group("Prompt configuration")
        prompt_group.add_argument("--prompt", type=str, required=(not supports_metadata_config), default=None, help="The textual description of the image to generate.")
        prompt_group.add_argument("--dual-prompts", "--dual_prompts", dest="dual_prompts", action="store_true", help="Enable dual prompts mode to use separate prompts for CLIP_L and T5 encoders.")
        prompt_group.add_argument("--clip_l-prompt", "--clip_l_prompt", dest="clip_l_prompt", type=str, default=None, help="The textual description for the CLIP_L encoder when dual prompts mode is enabled.")
        prompt_group.add_argument("--t5-prompt", "--t5_prompt", dest="t5_prompt", type=str, default=None, help="The textual description for the T5 encoder when dual prompts mode is enabled.")
        
        self.add_argument("--seed", type=int, default=None, nargs='+', help="Specify 1+ Entropy Seeds (Default is 1 time-based random-seed)")
        self.add_argument("--auto-seeds", type=int, default=0, help="Generate N random seed values automatically (creates N different images with different random seeds)")
        self._add_image_generator_common_arguments()
        if supports_metadata_config:
            self.add_metadata_config()

    def add_image_to_image_arguments(self, required=False) -> None:
        self.supports_image_to_image = True
        self.add_argument("--image-path", type=Path, required=required, default=None, help="Local path to init image")
        self.add_argument("--image-strength", type=float, required=False, default=ui_defaults.IMAGE_STRENGTH, help=f"Controls how strongly the init image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.IMAGE_STRENGTH})")

    def add_batch_image_generator_arguments(self) -> None:
        self.add_argument("--prompts-file", type=Path, required=True, default=argparse.SUPPRESS, help="Local path for a file that holds a batch of prompts.")
        self.add_argument("--global-seed", type=int, default=argparse.SUPPRESS, help="Entropy Seed (used for all prompts in the batch)")
        self._add_image_generator_common_arguments()

    def add_fill_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=True, help="Local path to the source image")
        self.add_argument("--masked-image-path", type=Path, required=True, help="Local path to the mask image")

    def add_depth_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=False, help="Local path to the source image")
        self.add_argument("--depth-image-path", type=Path, required=False, help="Local path to the depth image")
        self.add_argument("--save-depth-map", action="store_true", required=False, help="If set, save the depth map created from the source image.")

    def add_save_depth_arguments(self) -> None:
        self.add_argument("--image-path", type=Path, required=True, help="Local path to the source image")
        self.add_argument("--quantize",  "-q", type=int, choices=ui_defaults.QUANTIZE_CHOICES, default=None, required=False, help=f"Quantize the model ({' or '.join(map(str, ui_defaults.QUANTIZE_CHOICES))}, Default is None)")

    def add_redux_arguments(self) -> None:
        self.add_argument("--redux-image-paths", type=Path, nargs="*", required=True, help="Local path to the source image")

    def add_output_arguments(self) -> None:
        self.add_argument("--metadata", action="store_true", help="Export image metadata as a JSON file.")
        self.add_argument("--output", type=str, default="image.png", help="The filename for the output image. Default is \"image.png\".")
        self.add_argument('--stepwise-image-output-dir', type=str, default=None, help='[EXPERIMENTAL] Output dir to write step-wise images and their final composite image to. This feature may change in future versions.')
        self.add_argument('--stepwise-single-image', '--stepwise_single_image', dest="stepwise_single_image", action="store_true", help='[EXPERIMENTAL] When used with --stepwise-image-output-dir, creates a single image file that is updated at each step instead of separate files.')

    def add_image_outpaint_arguments(self, required=False) -> None:
        self.supports_image_outpaint = True
        self.add_argument("--image-outpaint-padding", type=str, default=None, required=required, help="For outpainting mode: CSS-style box padding values to extend the canvas of image specified by--image-path. E.g. '20', '50%%'")

    def add_controlnet_arguments(self) -> None:
        self.supports_controlnet = True
        self.add_argument("--controlnet-image-path", type=str, required=False, help="Local path of the image to use as input for controlnet.")
        self.add_argument("--controlnet-strength", type=float, default=ui_defaults.CONTROLNET_STRENGTH, help=f"Controls how strongly the control image influences the output image. A value of 0.0 means no influence. (Default is {ui_defaults.CONTROLNET_STRENGTH})")
        self.add_argument("--controlnet-save-canny", action="store_true", help="If set, save the Canny edge detection reference input image.")

    def add_metadata_config(self) -> None:
        self.supports_metadata_config = True
        self.add_argument("--config-from-metadata", "-C", type=Path, required=False, default=argparse.SUPPRESS, help="Re-use the parameters from prior metadata. Params from metadata are secondary to other args you provide.")

    def add_training_arguments(self) -> None:
        self.add_argument("--train-config", type=str, required=False, help="Local path of the training configuration file")
        self.add_argument("--train-checkpoint", type=str, required=False, help="Local path of the checkpoint file which specifies how to continue the training process")

    def parse_args(self, *args, **kwargs) -> argparse.Namespace:
        # Get arguments to parse - either from args parameter or from command line
        if args:
            # Use provided args for programmatic usage
            namespace = super().parse_args(*args, **kwargs)
        else:
            try:
                # Preprocess the arguments to handle backslash issues
                cli_args = sys.argv[1:]
                processed_args = []
                i = 0
                
                while i < len(cli_args):
                    arg = cli_args[i]
                    
                    # Check if this could be a backslash-related problem
                    if " --" in arg and not arg.startswith("--"):
                        # This could be a value followed by another argument due to backslash line continuation
                        parts = arg.split(" --", 1)
                        processed_args.append(parts[0])  # Add the value part
                        processed_args.append("--" + parts[1])  # Add the flag part
                    else:
                        # Just keep the argument as is
                        processed_args.append(arg)
                    
                    i += 1
                
                # Filter out any completely empty arguments
                filtered_args = [arg for arg in processed_args if arg.strip()]
                namespace = super().parse_args(filtered_args, **kwargs)
            except Exception as e:
                # Fall back to standard parsing if preprocessing fails
                filtered_args = [arg for arg in sys.argv[1:] if arg.strip()]
                namespace = super().parse_args(filtered_args, **kwargs)
            
            # Print debug output if requested
            if getattr(namespace, "debug", False):
                print("\n===== DEBUG: Command Arguments =====")
                
                # Build a map of dest->original_name from all arguments
                arg_name_mapping = {}
                for action in self._actions:
                    if action.dest != 'help':  # Skip the help action
                        # Find the longest option (most likely the primary name)
                        primary_option = max(action.option_strings, key=len) if action.option_strings else None
                        if primary_option:
                            # Store without the -- prefix
                            arg_name_mapping[action.dest] = primary_option.lstrip('-')
                
                # Print all args in the logical order they were defined in the parser
                # First, get important general args
                debug_order = [
                    'low-ram', 'debug', 'model', 'path', 'base-model', 'quantize'
                ]
                
                # Then add LoRA-related args
                lora_args = [
                    'lora-style', 'lora-paths', 'lora-scales', 'lora-name', 'lora-repo-id'
                ]
                debug_order.extend(lora_args)
                
                # Then prompt-related args
                prompt_args = [
                    'prompt', 'dual-prompts', 'clip_l-prompt', 't5-prompt'
                ]
                debug_order.extend(prompt_args)
                
                # Then generation parameters
                generation_args = [
                    'seed', 'auto-seeds', 'height', 'width', 'steps', 'guidance'
                ]
                debug_order.extend(generation_args)
                
                # Then image manipulation args
                image_args = [
                    'image-path', 'image-strength',
                ]
                debug_order.extend(image_args)
                
                # Then output args
                output_args = [
                    'metadata', 'output', 'stepwise-image-output-dir', 'stepwise-single-image'
                ]
                debug_order.extend(output_args)
                
                # Convert to set for faster lookups
                printed_keys = set()
                
                # Print in our specified order first
                for arg_name in debug_order:
                    # Find the corresponding key in the namespace
                    matching_key = None
                    for key, display_name in arg_name_mapping.items():
                        if display_name == arg_name:
                            matching_key = key
                            break
                    
                    if matching_key and hasattr(namespace, matching_key):
                        value = getattr(namespace, matching_key)
                        print(f"{arg_name}: {value}")
                        printed_keys.add(matching_key)
                
                # Print any remaining args not in our predefined order
                for key, value in vars(namespace).items():
                    if key not in printed_keys:
                        display_name = arg_name_mapping.get(key, key)
                        print(f"{display_name}: {value}")
                
                print("===================================\n")

        # Check if either training arguments are provided
        has_training_args = (hasattr(namespace, "train_config") and namespace.train_config is not None) or \
                            (hasattr(namespace, "train_checkpoint") and namespace.train_checkpoint is not None)

        # Only enforce model requirement for path if we're not in training mode
        if hasattr(namespace, "path") and namespace.path is not None and namespace.model is None and not has_training_args:
            self.error("--model must be specified when using --path")

        # Validate dual prompts configuration
        if getattr(namespace, "dual_prompts", False):
            clip_l_prompt = getattr(namespace, "clip_l_prompt", None)
            t5_prompt = getattr(namespace, "t5_prompt", None)
            
            # When dual prompts is enabled, always override the standard prompt to None
            # regardless of what might have been set in the command line
            namespace.prompt = None
            
            # Both prompts must be provided, though they can be empty strings
            if clip_l_prompt is None:
                self.error("--clip_l-prompt is required when dual prompts mode is enabled")
            if t5_prompt is None:
                self.error("--t5-prompt is required when dual prompts mode is enabled")

        if getattr(namespace, "config_from_metadata", False):
            prior_gen_metadata = json.load(namespace.config_from_metadata.open("rt"))

            if namespace.model is None:
                # when not provided by CLI flag, find it in the config file
                namespace.model = prior_gen_metadata.get("model", None)

            if namespace.base_model is None:
                namespace.base_model = prior_gen_metadata.get("base_model", None)

            # Only load prompt from metadata if dual prompts is not enabled
            if not (getattr(namespace, "dual_prompts", False)):
                if namespace.prompt is None:
                    namespace.prompt = prior_gen_metadata.get("prompt", None)

            # all configs from the metadata config defers to any explicitly defined args
            guidance_default = self.get_default("guidance")
            guidance_from_metadata = prior_gen_metadata.get("guidance")
            if namespace.guidance == guidance_default and guidance_from_metadata:
                namespace.guidance = guidance_from_metadata
            if namespace.quantize is None:
                namespace.quantize = prior_gen_metadata.get("quantize", None)
            seed_from_metadata = prior_gen_metadata.get("seed", None)
            if namespace.seed is None and seed_from_metadata is not None:
                namespace.seed = [seed_from_metadata]

            if namespace.seed is None:
                # not passed by user, not populated by metadata
                namespace.seed = [int(time.time())]

            if namespace.steps is None:
                namespace.steps = prior_gen_metadata.get("steps", None)

            if self.supports_lora:
                if namespace.lora_paths is None:
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", None)
                elif namespace.lora_paths:
                    # merge the loras from cli and config file
                    namespace.lora_paths = prior_gen_metadata.get("lora_paths", []) + namespace.lora_paths

                if namespace.lora_scales is None:
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", None)
                elif namespace.lora_scales:
                    # merge the loras from cli and config file
                    namespace.lora_scales = prior_gen_metadata.get("lora_scales", []) + namespace.lora_scales

            if self.supports_image_to_image:
                if namespace.image_path is None:
                    namespace.image_path = prior_gen_metadata.get("image_path", None)
                if namespace.image_strength == self.get_default("image_strength") and (img_strength_from_metadata := prior_gen_metadata.get("image_strength", None)):
                    namespace.image_strength = img_strength_from_metadata

            if self.supports_controlnet:
                if namespace.controlnet_image_path is None:
                    namespace.controlnet_image_path = prior_gen_metadata.get("controlnet_image_path", None)
                if namespace.controlnet_strength == self.get_default("controlnet_strength") and (cnet_strength_from_metadata := prior_gen_metadata.get("controlnet_strength", None)):
                    namespace.controlnet_strength = cnet_strength_from_metadata
                if namespace.controlnet_save_canny == self.get_default("controlnet_save_canny") and (cnet_canny_from_metadata := prior_gen_metadata.get("controlnet_save_canny", None)):
                    namespace.controlnet_save_canny = cnet_canny_from_metadata


            if self.supports_image_outpaint:
                if namespace.image_outpaint_padding is None:
                    namespace.image_outpaint_padding = prior_gen_metadata.get("image_outpaint_padding", None)

        # Only require model if we're not in training mode and require_model_arg is True
        if hasattr(namespace, "model") and namespace.model is None and not has_training_args and self.require_model_arg:
            self.error("--model / -m must be provided, or 'model' must be specified in the config file.")

        if self.supports_image_generation and namespace.seed is None and namespace.auto_seeds > 0:
            # choose N int seeds in the range of  0 < value < 1 billion
            namespace.seed = [random.randint(0, int(1e7)) for _ in range(namespace.auto_seeds)]
        elif self.supports_image_generation and namespace.seed is not None and namespace.auto_seeds > 0:
            # Warn user that --auto-seeds is being ignored
            print("\n⚠️  Warning: --auto-seeds is being ignored because --seed was specified. The --seed argument takes precedence.\n")

        if self.supports_image_generation and namespace.seed is None:
            # final default: did not obtain seed from metadata, --seed, or --auto-seeds
            namespace.seed = [int(time.time())]

        if self.supports_image_generation and len(namespace.seed) > 1:
            # auto append seed-$value to output names for multi image generations
            # e.g. output.png -> output_seed_101.png output_seed_102.png, etc
            output_path = Path(namespace.output)
            namespace.output = str(output_path.with_stem(output_path.stem + "_seed_{seed}"))

        if self.supports_image_generation and namespace.prompt is None and not (getattr(namespace, "dual_prompts", False)):
            # not supplied by CLI and not supplied by metadata config file and dual prompts not enabled
            self.error("--prompt argument required or 'prompt' required in metadata config file (or use dual prompts mode with --clip_l-prompt/--clip_l_prompt and --t5-prompt/--t5_prompt)")

        if self.supports_image_generation and namespace.steps is None:
            namespace.steps = ui_defaults.MODEL_INFERENCE_STEPS.get(namespace.model, 14)

        if self.supports_image_outpaint and namespace.image_outpaint_padding is not None:
            # parse and normalize any acceptable 1,2,3,4-tuple box value to 4-tuple
            namespace.image_outpaint_padding = box_values.parse_box_value(namespace.image_outpaint_padding)
            print(f"{namespace.image_outpaint_padding=}")

        return namespace
