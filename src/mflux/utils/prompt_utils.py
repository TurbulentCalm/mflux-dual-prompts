def normalize_dual_prompts(
    prompt: str | None,
    clip_prompt: str | None,
    t5_prompt: str | None,
    dual_prompt: bool
) -> tuple[str, str]:
    """
    Normalize prompts for dual or single prompt generation.

    Args:
        prompt: Single prompt fallback for both encoders if dual_prompt is False.
        clip_prompt: Optional CLIP-specific prompt.
        t5_prompt: Optional T5-specific prompt.
        dual_prompt: If True, use separate clip/t5 prompts. If False, fallback to prompt.

    Returns:
        A tuple of (clip_prompt, t5_prompt) with fallback logic applied.
    """
    if not dual_prompt:
        return prompt or "", prompt or ""
    return clip_prompt or "", t5_prompt or ""
