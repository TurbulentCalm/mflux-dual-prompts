# Developer Guide: Dual-Prompt Callback Refactor

## Overview
This document outlines the changes made to the callback system in order to support both single-prompt and dual-prompt configurations. These changes ensure that all callbacks continue to work with existing prompt inputs while enabling new generators to leverage `clip_prompt` and `t5_prompt`.

---

## 🔧 Refactored Files
The following files were updated:

- `callback.py`
- `callback_registry.py`
- `memory_saver.py`
- `stepwise_handler.py`
- `canny_saver.py`
- `depth_saver.py`

Each of these now supports:
```python
prompt: str | None = None,
dual_prompt: bool = False,
clip_prompt: str | None = None,
t5_prompt: str | None = None
```

---

## ✅ Normalized Fallback Logic
At the top of every `call_*` method, fallback logic is included:

```python
if not dual_prompt:
    clip_prompt = prompt or ""
    t5_prompt = prompt or ""
else:
    clip_prompt = clip_prompt or ""
    t5_prompt = t5_prompt or ""
```

This ensures:
- Legacy generators using `prompt` continue to work.
- Dual-prompt generators receive valid strings.

---

## 🔌 Compatibility Strategy
- No interface-breaking changes were introduced.
- Existing `call_before_loop(prompt=...)` signatures now silently accept additional keyword arguments.
- All callback registration via `CallbackRegistry` works as before.

---

## 📦 Best Practices for New Callbacks
When writing new callbacks:
- Always declare `dual_prompt`, `clip_prompt`, and `t5_prompt` in your `call_*` signatures.
- Use the same fallback pattern shown above.

---

## 🧪 Testing
If you are manually testing callback compatibility:
- Use both single-prompt and dual-prompt scripts
- Ensure metadata and saving behaviors match expectations

---

## 🧠 Final Thoughts
This is a flexible approach that keeps backward compatibility, avoids branching in base code, and aligns with the new prompt strategy introduced in `ImageGenerator`.

If upstream maintainers are unfamiliar with this pattern, direct them to this document.

---

*Last updated: May 16, 2025*
