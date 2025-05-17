def normalize_dual_prompts(
    dual_prompts: bool,
    prompt: str | None,
    clip_prompt: str | None,
    t5_prompt: str | None
) -> tuple[str, str]:
    """
    Normalize prompts for dual or single prompt generation.

    Args:
        dual_prompts: If True, use separate clip/t5 prompts. If False, fallback to prompt.
        prompt: Single prompt fallback for both encoders if dual_prompts is False.
        clip_prompt: Optional CLIP-specific prompt.
        t5_prompt: Optional T5-specific prompt.

    Returns:
        A tuple of (clip_prompt, t5_prompt) with fallback logic applied.
    """
    if not dual_prompts:
        return prompt or "", prompt or ""
    return clip_prompt or "", t5_prompt or ""