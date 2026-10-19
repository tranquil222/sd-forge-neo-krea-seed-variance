# Krea Seed Variance Enhancer for Forge Neo

An unofficial, standalone port of [HARUKI's KreaSeedVarianceEnhancer](https://github.com/harukimix/KreaSeedVarianceEnhancer).
Adds seed-dependent noise to Krea 2's positive text conditioning. It does not load,
edit, merge, scale, or replace any LoRA. Your LoRA's influence on the resulting image
can nevertheless change when conditioning changes.

## Install / use

In **Extensions → Install from URL**, enter:

```text
https://github.com/tranquil222/sd-forge-neo-krea-seed-variance
```

Install, then **restart Forge completely**. Alternatively, clone the repository
under Forge Neo's `extensions` directory. If you already have a manually installed
copy, replace that copy rather than keeping two copies active.
No additional packages or model downloads are needed. Open **Krea Seed Variance
Enhancer** under txt2img. It starts disabled.

1. Turn **off** the original **Seed Variance Enhancer** panel. If both are enabled,
   this port skips itself and reports why; it never silently overrides the original.
2. Enable **Krea Seed Variance Enhancer**. Defaults match the original node's main
   settings: auto strength 1, randomize 50%, Beginning, switch at 25%, uniform, values.
3. Use **Follow image seed** for normal seed-to-seed variation. To keep the image seed
   fixed while exploring conditioning variations, use **Random per image**, or change
   the independent **Fixed variation seed**.
4. Read the console or expand **Diagnostics / credits → Refresh diagnostics** after
   generation. It reports measured embedding scale, actual perturbation, and active
   model evaluations. Those evaluations are not the displayed sampler step count.

The original node was tuned for Krea 2 Turbo. This port accepts Forge Neo's Krea 2
RAW/Turbo engine, including RAW with a Turbo LoRA, but visual results at your LoRA
strength and sampler settings need testing. It cannot guarantee different poses,
preserved identity, or better texture.

## Easy presets

Choose **Easy preset** at the top of the panel to fill the effect controls. You can
then fine-tune anything; the dropdown shows **Custom / current settings** when the
values no longer match a recipe. These are convenience recipes, not visually
validated quality guarantees for RAW + Turbo LoRA.

| Preset | Auto strength | Randomize | Timing | Selection |
| --- | --- | --- | --- | --- |
| Light | 0.5 | 30% | All | Gaussian / tokens |
| Light+ | 0.75 | 30% | All | Gaussian / tokens |
| Medium | 1 | 50% | All | Gaussian / tokens |
| Medium+ | 1.5 | 50% | All | Gaussian / tokens |
| Heavy | 2 | 65% | All | Gaussian / tokens |
| Heavy+ | 3 | 65% | All | Gaussian / tokens |
| Chaos | 4 | 80% | All | Gaussian / tokens |
| Chaos+ | 6 | 80% | All | Gaussian / tokens |
| Early exploration | 1 | 50% | Beginning, switch at 40% progress | Gaussian / tokens |
| Late variation | 0.5 | 30% | Ending, switch at 65% progress | Gaussian / values |
| Protect prompt opening | 1 | 50% | All; protect first 40% of tokens | Gaussian / tokens |
| Original defaults | 1 | 50% | Beginning, switch at 25% progress | Uniform / values |

Each **+** keeps its base preset's settings and increases only auto strength by
50%. **Medium+** is the 1.5 / 50% / All / Gaussian / tokens setup reported useful
with Krea 2 RAW + Turbo LoRA; results remain setup-dependent.

All recipes reset manual strength to 20 (inactive because auto strength is on),
protection to 0% except Protect prompt opening, and the unused All-mode switch
value to 25%. Timing percentages are sigma-based progress, not counted steps.
The preset **does not** toggle Enable, change either seed control, change logging,
adjust LoRAs, or touch any other generation setting. To get fresh conditioning
variations with the same image seed, choose **Random per image** separately.

PNG metadata includes the matching preset name for reference, but replay always
restores the actual saved controls, not the recipe. IIB's normal **Send to txt2img**
uses the same restore path. The helper dropdown is not a generation input, so
existing queue snapshots retain the same input layout and values.

## Controls

| Control | Meaning |
| --- | --- |
| Auto strength factor | Measured embedding standard deviation × factor. 0 uses Manual strength. |
| Randomize (%) | Fraction of values/tokens selected for noise. |
| Beginning / Ending / All | Use noisy conditioning before/after the switch, or throughout sampling. |
| Switch at (% denoising progress) | Model sigma-based progress, matching the upstream mechanism. Not a percentage of counted steps. Ignored for All. |
| Variation seed mode | Follow image seed, independent Fixed seed, or Random per image. |
| Noise selection | Individual feature values or complete encoded token vectors. Tokens are not necessarily whole words. |
| Noise distribution | Uniform or Gaussian, matched for expected power. |
| Protect prompt | Leave a prefix/suffix of encoded tokens unchanged. Does not isolate/protect a particular LoRA or subject. |

Beginning deliberately restores clean conditioning later, so the composition can
settle back toward the model's usual result. **All** keeps the conditioning variation
present, with a greater risk of losing prompt adherence. No sampler can be assumed
to preserve an early preview. Start by comparing one setting at a time at fixed
image and variation seeds; large strengths can produce artifacts.

## PNG Info and queued jobs

All controls, version, sigma-based timing, actual per-image variation seed, and
application/skip status are saved in Forge's standard PNG generation parameters
when PNG metadata saving is enabled. **PNG Info → Send to txt2img** restores them
when this extension is installed. Random-per-image images restore **Fixed** mode
using their actual saved seed; otherwise replay would randomize again. PNGs without
Krea SVE metadata turn this extension off when pasted.

Queues that capture native extension controls can capture these settings with each
job. Queue compatibility depends on the queue extension; no queue is required here.

Use the original PNG with embedded parameters. IIB's normal **Send to txt2img**
loads these controls; **send only prompt** does not. Match the checkpoint, LoRAs
and their weights, image seed, sampler/scheduler, steps, CFG and other extensions
as well. Loading this extension's settings alone cannot recreate missing dependencies.

For images that also use Sigma Rescale Full, install
[Sigma Rescale Neo](https://github.com/tranquil222/sd-forge-neo-sigma-rescale),
which imports the Full and Simple metadata, including late-start percentages.
See [the copy-ready Civitai note](docs/CIVITAI.md).

## Scope / removal

Main txt2img pass only. Hires conditioning is untouched; refiners and other model
families are explicitly skipped. RES4LYF and ordinary KDiffusion samplers use the
same Forge conditioning callback; internal substeps are gated by their actual sigma,
not a guessed step counter. Sigma Rescale can change which evaluations fall inside
the selected window.

No Forge source files are patched. Disable this extension in Extensions or remove
its folder and restart to remove it. As with other extensions that add controls,
removing it can invalidate jobs captured while it was installed; finish those first.

Tests run with Forge's Python:

```powershell
..\..\venv\Scripts\python.exe -m unittest discover -s tests -v
```

For a checkout outside Forge, set `FORGE_ROOT` to your Forge directory to enable
the real PNG formatter/parser checks. `KREA_SVE_UPSTREAM` can point to the pinned
upstream Python file for numerical parity checks. These are CPU tests; no models
are loaded. Version **0.1.2** has 35 passing checks with both references present,
plus 17 isolated browser checks of presets and restoration. Full-model visual
results are setup-dependent; this remains a preview release.

See [UPSTREAM.md](UPSTREAM.md) and [LICENSE](LICENSE) for provenance and port differences.
