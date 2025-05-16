# Refactor Log: ImageGenerator Integration

## Summary
Refactored the repeated image generation logic across `generate_*.py` scripts into a single reusable class named `ImageGenerator`. This improves maintainability, consistency, and reusability.

---

## ✅ Files Created

### `image_generator.py`
- Encapsulates:
  - Callback setup
  - Model instantiation
  - Seed loop
  - Image generation
  - Image saving
  - Memory stats
- Designed to accept any compatible `Flux1*` model class.

### `generate_refactored.py`
- A refactored version of `generate.py`
- Delegates to `ImageGenerator`
- Significantly shorter and more maintainable

---

## 🧩 Logic Extracted From
- CLI argument parsing and dispatch
- Model creation using `ModelConfig.from_name()`
- Per-seed image generation loop
- Output path formatting
- Optional JSON metadata
- Memory saver reporting

---

## 💡 Next Steps
- Refactor other `generate_*.py` files (`generate_controlnet.py`, `generate_depth.py`, etc.) to use `ImageGenerator`
- Extend `ImageGenerator` if variant-specific behavior is needed (via `extra_args_func` or subclassing)

---

*Logged: May 16, 2025*
