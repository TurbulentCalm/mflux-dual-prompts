import mlx.core as mx

from mflux.utils.prompt_utils import normalize_dual_prompts
from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class PromptEncoder:
    @staticmethod
    def encode_prompts(
        prompt: str | None,
        clip_prompt: str | None,
        t5_prompt: str | None,
        dual_prompt: bool,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: TokenizerT5,
        clip_tokenizer: TokenizerCLIP,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
    ) -> tuple[mx.array, mx.array]:
        """
        Encodes prompts using both T5 and CLIP tokenizers and text encoders.

        If dual_prompt is False:
            - Both clip_prompt and t5_prompt fall back to the shared prompt value.
        If dual_prompt is True:
            - clip_prompt and t5_prompt are used independently, with empty string fallback.

        This function handles prompt fallback logic internally, so callers can pass raw values without preprocessing.

        Args:
            prompt: The shared prompt used when dual_prompt is False.
            clip_prompt: Optional clip-specific prompt (used if dual_prompt is True).
            t5_prompt: Optional T5-specific prompt (used if dual_prompt is True).
            dual_prompt: Whether to use separate clip/t5 prompts.
            prompt_cache: Cache dictionary for avoiding redundant encoding.
            t5_tokenizer: T5 tokenizer instance.
            clip_tokenizer: CLIP tokenizer instance.
            t5_text_encoder: T5 text encoder instance.
            clip_text_encoder: CLIP text encoder instance.

        Returns:
            Tuple of (T5 encoded embeddings, CLIP pooled embeddings)
        """

        # --- Fallback Logic ---
        clip_prompt, t5_prompt = normalize_dual_prompts(prompt, clip_prompt, t5_prompt, dual_prompt)

        # --- Caching ---
        cache_key = (clip_prompt, t5_prompt)
        if cache_key in prompt_cache:
            return prompt_cache[cache_key]

        # --- Tokenization ---
        t5_tokens = t5_tokenizer.tokenize(t5_prompt)
        clip_tokens = clip_tokenizer.tokenize(clip_prompt)

        # --- Encoding ---
        t5_embeddings = t5_text_encoder(t5_tokens)
        clip_embeddings = clip_text_encoder(clip_tokens)

        # --- Caching Encoded Result ---
        prompt_cache[cache_key] = (t5_embeddings, clip_embeddings)

        return t5_embeddings, clip_embeddings