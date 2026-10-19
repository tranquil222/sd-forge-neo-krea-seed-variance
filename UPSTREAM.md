# Upstream and port notes

Unofficial Forge Neo adaptation of [HARUKI / KreaSeedVarianceEnhancer](https://github.com/harukimix/KreaSeedVarianceEnhancer).

- Pinned commit: `1515d23a5b399a44ccd97482e1105a43937268ae`.
- Upstream code version: `k2-1.2.0 (base: SVE 2.2)`.
- Original SeedVarianceEnhancer: [ChangeTheConstants](https://github.com/ChangeTheConstants/SeedVarianceEnhancer), v2.2.
- License: MIT No Attribution (MIT-0). See LICENSE for the upstream notice and terms. The Forge adaptation uses the same terms.

Preserved: measured fp32 embedding standard deviation; auto/manual strength;
power-matched uniform/Gaussian noise; independent selection seed; value/token
selection; protected prefix/suffix; null-token protection; deterministic noise;
legacy manual-strength seed-reset encoding; positive conditioning only.

Forge adaptations:

- Flatten `[tokens, 12, 2560]` into `[1, tokens, 30720]` while generating noise,
  then restore the twelve-layer layout. Encoder layers are not prompt tokens.
- Local device generators instead of saving/resetting global Torch RNG state.
- Per-image seeds and per-branch prompt schedules; never discard branches.
- Noise is computed once per distinct scheduled embedding and seed, then reused.
  Clean cached embeddings are never edited. Forge-added batch padding gets zero noise.
- Use the model predictor's `percent_to_sigma()` to match upstream diffusion-progress
  timing. This is NOT a fixed percentage of outer steps or model evaluations.
  At the exact boundary this port selects one branch (clean for Beginning, noisy
  for Ending); upstream's overlapping conditioning ranges may average both there.
- Per-image PNG settings and actual seeds, a manual diagnostics viewer, and console
  diagnostics. Random-mode PNGs restore Fixed mode for replay.
- Empty/all-padding statistics are finite instead of NaN. Non-finite/unsupported
  conditioning is skipped with an explicit message and PNG status.
- First release supports Krea 2 txt2img main passes only, not Hires/refiners/img2img.
  No monkey patches, install-time dependencies, downloads, or model-weight changes.

This does not promise pixel-identical ComfyUI output: text encoding, samplers,
padding, devices/dtypes, and exact-boundary behavior can differ. Upstream was tuned
for Krea 2 Turbo, not specifically RAW + a partial-strength Turbo LoRA.
