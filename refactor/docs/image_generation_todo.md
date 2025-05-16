# TODO: Image Generation Refactor and Improvements

## 🧩 Refactor Opportunities

### 1. Consolidate generation logic into a shared class
- **Proposed class name**: `ImageGenerator` (preferred) or `ImageGenerationPipeline`
- **Purpose**: Eliminate repeated logic across `generate.py`, `generate_depth.py`, `generate_controlnet.py`, etc.
- **Functionality**:
  - Wraps `generate_image()` calls
  - Handles prompt routing (dual/single)
  - Applies consistent saving and metadata logic
  - Optionally handles memory stats and callbacks

---

## 🔄 Centralize Prompt Handling
- All fallback logic for `prompt`, `clip_prompt`, and `t5_prompt` is now handled **inside** `PromptEncoder.encode_prompts()`.
- **Developer rule**: do not normalize prompts outside this method
- **Rationale**: Avoids duplication and enforces a single source of truth

---

## 📦 Housekeeping and Usability

### 1. Add inline TODOs as comments in markdown
- Use `<!-- TODO: ... -->` for hidden notes
- Use `> ⚠️ TODO:` for visible callouts during draft/review

### 2. Docstring and code comments
- Add explicit docstring to `PromptEncoder.encode_prompts()` describing fallback logic
- Add a `# central fallback logic` inline comment for future contributors

---

## ✨ Future Improvements (Suggestions)

### 1. Add timing/profiling inside generator loop
- Could be logged or optionally printed

### 2. Refactor CLI argument parsing boilerplate
- Extract common `argparse` logic into a shared parser helper
- Optional: convert into `click`-based CLI framework if needed

### 3. Abstract metadata saving
- Standardize metadata output handling inside the generator class
- Allow optional suppression or redirect to a log file

---

*Last updated: May 16, 2025*
