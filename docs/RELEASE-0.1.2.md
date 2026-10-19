# 0.1.2 preview — Forge Neo port with easy presets

Unofficial Forge Neo port of HARUKI's KreaSeedVarianceEnhancer, distributed under
the upstream MIT-0 license with credits and pinned-source notes.

- Krea 2 RAW/Turbo positive txt2img conditioning, including RAW + Turbo LoRA.
- Light, Medium, Heavy, Chaos and matching **+** variants. Medium+ is auto 1.5,
  randomize 50%, All, Gaussian, tokens, no protected tokens.
- Early exploration, Late variation, Protect prompt opening, Original defaults.
- Presets change effect controls only, not Enable, seeds, logging or LoRA weights.
- Per-image PNG settings and actual variation seeds, restored by PNG Info and IIB.
- Random-per-image metadata restores Fixed mode for reproducibility.
- Original generic SVE conflict detection, explicit skip status and diagnostics.
- No Forge source edits, extra packages, model downloads or LoRA-file changes.

35 CPU tests passed with the pinned upstream and Forge references, including 72
upstream tensor comparisons and real Forge PNG formatting/parsing. Seventeen
isolated browser checks passed for presets, manual tweaks, PNG restoration and
narrow layouts. Full-model quality and pixel equality across hosts are not claimed.

Install using **Extensions → Install from URL**, then restart Forge completely.
Disable the original generic Seed Variance Enhancer while using the Krea port.
If updating a manually installed copy, replace it; do not keep duplicate copies.

See [README](../README.md) for installation and [the Civitai note](CIVITAI.md)
for sharing both this and Sigma Rescale Neo with your images.
