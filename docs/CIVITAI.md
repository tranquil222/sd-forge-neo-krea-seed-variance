# Copy-ready Civitai note

Some of my images use **Krea Seed Variance Enhancer** and **Sigma Rescale** in Forge Neo.
You can install the tools here:

- Krea Seed Variance Enhancer: https://github.com/tranquil222/sd-forge-neo-krea-seed-variance
- Sigma Rescale Neo (loads my Simple/Full settings): https://github.com/tranquil222/sd-forge-neo-sigma-rescale

In Forge, open **Extensions → Install from URL**, install each extension you need,
then restart. Load the original PNG using **PNG Info → Send to txt2img**, or IIB's
normal **Send to txt2img**, to restore the saved extension settings. The Krea preset
name is just a shortcut; replay uses the actual saved values and variation seed.
Sigma Neo also restores Full's **Begin applying at (% of steps)** setting.

You still need the same model, LoRAs/weights, sampler, scheduler and other generation
settings. Screenshots or recompressed images may not include the metadata. Disable
the original generic Seed Variance Enhancer when using the Krea port, and don't
enable old Sigma Simple/Full alongside Neo.

The Krea extension is an unofficial Forge Neo port of
[HARUKI's KreaSeedVarianceEnhancer](https://github.com/harukimix/KreaSeedVarianceEnhancer).
Sigma Neo is an independently written tool inspired by RES4LYF/lz-55's workflow.
Neither changes your LoRA files or weight settings; changing conditioning or the
sigma schedule can still change the resulting image.
