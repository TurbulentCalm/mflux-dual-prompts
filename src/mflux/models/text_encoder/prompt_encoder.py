import mlx.core as mx

from mflux.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from mflux.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from mflux.tokenizer.clip_tokenizer import TokenizerCLIP
from mflux.tokenizer.t5_tokenizer import TokenizerT5


class PromptEncoder:
    @staticmethod
    def encode_prompts(
        clip_prompt: str | None,
        t5_prompt: str | None,
        prompt_cache: dict[str, tuple[mx.array, mx.array]],
        t5_tokenizer: TokenizerT5,
        clip_tokenizer: TokenizerCLIP,
        t5_text_encoder: T5Encoder,
        clip_text_encoder: CLIPEncoder,
    ) -> tuple[mx.array, mx.array]:

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