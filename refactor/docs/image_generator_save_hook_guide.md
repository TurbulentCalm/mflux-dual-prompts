# Developer Guide: Custom Save Logic in ImageGenerator

## Overview
The `ImageGenerator` class now supports a subclassing mechanism that allows different `generate_*.py` scripts to customize how images are saved after generation.

This resolves the problem of differing output logic (e.g., `get_right_half()` in `generate_in_context.py`) without requiring duplication of the generation loop or breaking the abstraction.

---

## ✅ The Problem
Some image generators (like InContext) produce composite images, only part of which should be saved by default. Others do not. Previously, these save routines were commented out or duplicated, making maintenance difficult.

---

## ✅ The Solution
We added a **`save_image(self, image, seed, args)`** method to the `ImageGenerator` class. This is now called at the end of the loop inside `run()`.

### Default behavior (in base class):
```python
image.save(path=args.output.format(seed=seed), export_json_metadata=args.metadata)
```

---

## How to Use This Feature

### 1. Subclass `ImageGenerator` for your script
```python
class InContextImageGenerator(ImageGenerator):
    def save_image(self, image, seed, args):
        output_path = Path(args.output.format(seed=seed))
        image.get_right_half().save(path=output_path, export_json_metadata=args.metadata)
        if getattr(args, "save_full_image", False):
            image.save(path=output_path.with_stem(output_path.stem + "_full"))
```

### 2. Use this subclass in your script
```python
generator = InContextImageGenerator(model_class=Flux1InContextLoRA)
generator.run(args)
```

### 3. That’s it! The `run()` method takes care of everything else.

---

## Why Not Use If-Statements?
We wanted to avoid cluttering `ImageGenerator` with special-case logic or flags like `if args.tool == "in_context"`. Using inheritance is cleaner, scalable, and easier to read.

---

## When to Use Subclassing
Use subclassing if:
- You need to conditionally save different image formats.
- You want to skip saving under certain circumstances.
- You have additional post-processing tied to saving.

---

## Notes for Upstream Maintainers
This refactor introduces a lightweight object-oriented structure into otherwise procedural CLI scripts. This was done:
- To reduce code duplication across generator variants.
- To isolate special-case logic in a clean, overrideable way.
- Without disrupting the behavior of any existing generator.

If you're unfamiliar with subclassing, treat `ImageGenerator` like a framework with extension points.

---

*Last updated: May 16, 2025*
