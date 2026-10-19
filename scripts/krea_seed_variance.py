"""Standalone Forge Neo adapter; no model/LoRA or Forge source patching."""
from pathlib import Path
from types import SimpleNamespace
import secrets
import sys
import threading

import gradio as gr
import torch
from modules import scripts, script_callbacks
from modules.infotext_utils import PasteField

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from krea_sve_core import (VERSION, TITLE, PREFIX, KEYS, DEFAULTS, LABELS, RESTORERS,
                           WINDOWS, SEED_MODES, as_bool, validate_options, conditioning_noise,
                           window_active, selected_entry, stack_noise)
from krea_sve_presets import (CUSTOM, CHOICES, PRESET_KEYS, preset_values,
                              matching_preset, preset_description)

ATTR = "_krea_sve_state"
_LOCK = threading.Lock()
_LAST = "No Krea SVE generation in this session. Enable the panel, then generate."


def diagnostics_text():
    with _LOCK:
        return _LAST


def apply_preset(name):
    values = preset_values(name)
    return [values[key] if values is not None else gr.skip() for key in PRESET_KEYS]


def identify_preset(*values):
    name = matching_preset(dict(zip(PRESET_KEYS, values)))
    return name, preset_description(name)


def report(text, log=False):
    global _LAST
    with _LOCK:
        _LAST = text
    if log:
        print("[Krea SVE] " + text, flush=True)


def skip(p, state, reason):
    state.active = False
    state.schedules = []
    state.options = None
    p.extra_generation_params[PREFIX+" status"] = "Skipped: " + reason
    report("Skipped: " + reason, True)
    try:
        gr.Warning("Krea SVE skipped: " + reason)
    except Exception:
        pass


def image_index(p, index, all_seeds):
    # Forge uses batch-local seeds while saving, and all_seeds for result replay.
    return index + p.iteration * p.batch_size if all_seeds is getattr(p, "seeds", None) else index


def metadata_seed(p, index=0, all_seeds=None, **kwargs):
    state = getattr(p, ATTR)
    return str(state.seeds[image_index(p, index, all_seeds)])


def legacy_sve_enabled(p):
    # This is set by the existing enhancer after XYZ and per-batch overrides.
    return as_bool(p.extra_generation_params.get("SVE Enable", False))


@torch.inference_mode()
def on_cfg(params):
    denoiser = params.denoiser
    p = getattr(denoiser, "p", None)
    state = getattr(p, ATTR, None)
    if state is None or not state.active or getattr(p, "is_hr_pass", False):
        return
    try:
        if getattr(denoiser, "_refiner_pass", False):
            skip(p, state, "Refiner pass is not supported")
            return
        if legacy_sve_enabled(p):
            skip(p, state, "Turn off the original Seed Variance Enhancer; the two do not stack")
            return
        predictor = p.sd_model.forge_objects.unet.model.predictor
        boundary = float(predictor.percent_to_sigma(state.options["switch_percent"] / 100))
        # All batch members share one timestep in Forge. No counters inferred
        # from sampling_step or denoiser.step: RES samplers evaluate substeps.
        if not window_active(state.options["window"], params.sigma.flatten()[0].item(), boundary):
            return
        rows = [selected_entry(schedule, denoiser.step).cond for schedule in state.schedules]
        delta = stack_noise(rows, params.text_cond)
        params.text_cond = params.text_cond + delta
        state.calls += 1
        p.extra_generation_params[PREFIX+" status"] = "Applied (main pass)"
    except Exception as exc:
        skip(p, state, str(exc))


class KreaSeedVariance(scripts.Script):
    sorting_priority = 1130

    def title(self):
        return TITLE

    def show(self, is_img2img):
        return False if is_img2img else scripts.AlwaysVisible

    def ui(self, is_img2img):
        controls = {}
        def args(key, label, info=None):
            return dict(value=DEFAULTS[key], label=label, info=info, elem_id="krea_sve_txt2img_"+key)
        with gr.Accordion(TITLE, open=False, elem_id="krea_sve_txt2img_panel"):
            controls["enabled"] = gr.Checkbox(**args("enabled", "Enable Krea Seed Variance Enhancer"))
            preset = gr.Dropdown(CHOICES, value=CUSTOM, label="Easy preset", filterable=False,
                info="Fills the effect controls below. Enable, seeds and logging stay as chosen. These are starting recipes, not guaranteed results.",
                elem_id="krea_sve_txt2img_preset")
            preset.do_not_save_to_config = True
            description = gr.Markdown(preset_description(CUSTOM), elem_id="krea_sve_txt2img_preset_description")
            gr.Markdown("Krea 2 positive-prompt conditioning only. **Turn off the original Seed Variance Enhancer.** "
                        "Your LoRA files and weight settings stay unchanged; identity/style can still vary. "
                        "Main txt2img pass only; Hires is unchanged.")
            with gr.Row():
                controls["auto_strength_factor"] = gr.Slider(0, 100, step=0.05, **args("auto_strength_factor", "Auto strength factor",
                    "Noise scale = measured embedding standard deviation × factor. 0 uses manual strength."))
                controls["randomize_percent"] = gr.Slider(1, 100, step=1, **args("randomize_percent", "Randomize (%)",
                    "Percentage of values or tokens selected for noise; not a pose-only control."))
            with gr.Row():
                controls["window"] = gr.Dropdown(WINDOWS, **args("window", "Apply noise during",
                    "Beginning restores the clean prompt later. All keeps the variation active throughout."))
                controls["switch_percent"] = gr.Slider(1, 99, step=1, **args("switch_percent", "Switch at (% denoising progress)",
                    "Matches the original node's sigma-based timing, NOT a percentage of counted steps. Ignored for All."))
            with gr.Row():
                controls["seed_mode"] = gr.Dropdown(SEED_MODES, **args("seed_mode", "Variation seed mode",
                    "Follow image seed varies with each image. Random per image also varies when the image seed stays fixed."))
                controls["seed"] = gr.Textbox(**args("seed", "Fixed variation seed",
                    "Independent of the image seed; used only in Fixed mode. PNGs store the actual variation seed."))
            with gr.Accordion("Advanced", open=False):
                with gr.Row():
                    controls["distribution"] = gr.Dropdown(["uniform", "gaussian"], **args("distribution", "Noise distribution",
                        "Gaussian and uniform are matched for expected noise power."))
                    controls["granularity"] = gr.Dropdown(["values", "tokens"], **args("granularity", "Noise selection",
                        "Values selects individual features. Tokens selects whole token vectors across all 12 encoder layers."))
                controls["strength"] = gr.Number(**args("strength", "Manual strength", "Used only when Auto strength factor is 0."))
                with gr.Row():
                    controls["protect_from"] = gr.Dropdown(["beginning", "end"], **args("protect_from", "Protect prompt from"))
                    controls["protect_percent"] = gr.Slider(0, 99, step=1, **args("protect_percent", "Protect prompt (%)",
                        "Protects this proportion of encoded tokens, not particular subjects or LoRAs."))
                controls["log"] = gr.Checkbox(**args("log", "Print diagnostics to console"))
            with gr.Accordion("Diagnostics / credits", open=False):
                diagnostics = gr.Textbox(value=diagnostics_text, label="Last generation diagnostics", lines=5, interactive=False)
                gr.Button("Refresh diagnostics", size="sm").click(diagnostics_text, [], diagnostics, queue=False)
                gr.Markdown("Unofficial Forge Neo port of [HARUKI's KreaSeedVarianceEnhancer]"
                            "(https://github.com/harukimix/KreaSeedVarianceEnhancer), based on "
                            "ChangeTheConstants' SeedVarianceEnhancer. MIT-0. "
                            "Defaults follow the original node; RAW + Turbo LoRA results need visual testing. "
                            "PNG Info → Send to txt2img restores these controls. Random-mode PNGs restore a fixed seed for replay.")
        effect_controls = [controls[key] for key in PRESET_KEYS]
        # Only a user's selection applies a recipe. Programmatic changes from
        # PNG restore, startup or matching must never reapply/overwrite values.
        preset.input(apply_preset, [preset], effect_controls, queue=False, show_progress="hidden")
        for control in effect_controls:
            control.change(identify_preset, effect_controls, [preset, description],
                           queue=False, show_progress="hidden")
        gr.context.Context.root_block.load(identify_preset, effect_controls, [preset, description],
                                          queue=False, show_progress="hidden")
        self.infotext_fields = [PasteField(controls[key], RESTORERS[key]) for key in KEYS]
        self.paste_field_names = list(LABELS.values())
        return [controls[key] for key in KEYS]

    def process(self, p, *values):
        state = SimpleNamespace(active=False, options=None, schedules=[], seeds=[], calls=0)
        setattr(p, ATTR, state)
        for key in list(p.extra_generation_params):
            if key == PREFIX or key.startswith(PREFIX+" "):
                p.extra_generation_params.pop(key)
        supplied = dict(DEFAULTS, **dict(zip(KEYS, values)))
        if not as_bool(supplied["enabled"]):
            return
        try:
            state.options = validate_options(supplied)
            mode = state.options["seed_mode"]
            state.seeds = [int(seed) % (1 << 64) if mode == "Follow image seed" else
                           secrets.randbits(64) if mode == "Random per image" else int(state.options["seed"])
                           for seed in p.all_seeds]
            p.extra_generation_params.update({LABELS[key]: state.options[key] for key in KEYS})
            p.extra_generation_params[PREFIX+" version"] = VERSION
            p.extra_generation_params[PREFIX+" preset"] = matching_preset(state.options)
            p.extra_generation_params[LABELS["seed"]] = metadata_seed
            p.extra_generation_params[PREFIX+" timing"] = "sigma progress"
            p.extra_generation_params[PREFIX+" status"] = "Not applied"
        except Exception as exc:
            skip(p, state, str(exc))

    @torch.inference_mode()
    def process_before_every_sampling(self, p, *values, **kwargs):
        state = getattr(p, ATTR, None)
        if state is None:
            return
        state.active, state.schedules = False, []
        if state.options is None or getattr(p, "is_hr_pass", False):
            return
        state.calls = 0
        try:
            if type(p.sd_model).__name__ != "Krea2" or type(p.sd_model).__module__ != "backend.diffusion_engine.krea":
                raise ValueError("This port supports Krea 2 models only")
            if legacy_sve_enabled(p):
                raise ValueError("Turn off the original Seed Variance Enhancer; the two do not stack")
            if getattr(p, "refiner_checkpoint_info", None) is not None:
                raise ValueError("Disable the refiner for Krea SVE")
            cond = kwargs["c"]
            offset = p.iteration * p.batch_size
            # Other scripts may finalize seeds/batch counts after process().
            # Follow the actual sampling batch, not an early UI snapshot.
            if len(p.seeds) != len(cond.batch):
                raise ValueError("Prompt/seed batch sizes do not match")
            mode = state.options["seed_mode"]
            for i, image_seed in enumerate(p.seeds):
                index = offset+i
                if index >= len(state.seeds):
                    if index != len(state.seeds):
                        raise ValueError("Non-contiguous image batch")
                    state.seeds.append(secrets.randbits(64) if mode == "Random per image" else int(state.options["seed"]))
                if mode == "Follow image seed":
                    state.seeds[index] = int(image_seed) % (1 << 64)
            seeds = state.seeds[offset:offset+len(cond.batch)]
            if len(seeds) != len(cond.batch):
                raise ValueError("Prompt/seed batch sizes do not match")
            prepared, lines = [], []
            # Deduplicate repeated scheduled tensors without editing cached conds.
            cache = {}
            for image, (branches, seed) in enumerate(zip(cond.batch, seeds)):
                for branch in branches:
                    schedule = []
                    for entry in branch.schedules:
                        emb = entry.cond
                        if not isinstance(emb, torch.Tensor) or emb.ndim != 3 or emb.shape[-2:] != (12, 2560):
                            raise ValueError("Expected Krea 2 [tokens, 12, 2560] conditioning")
                        key = (id(emb), seed)
                        if key not in cache:
                            delta, diag = conditioning_noise(emb.reshape(1, emb.shape[0], -1), state.options, seed)
                            cache[key] = delta.reshape_as(emb)
                            lines.append(f"Image {offset+image+1}: " + diag.text())
                        schedule.append(SimpleNamespace(end_at_step=entry.end_at_step, cond=cache[key]))
                    if not schedule:
                        raise ValueError("Empty prompt schedule")
                    prepared.append(schedule)
            state.schedules, state.active = prepared, bool(prepared)
            state.summary = "\n".join(lines)
            # PNG replay must not leave an already-checked legacy SVE enabled.
            # The conflict check above happens first; this changes metadata only.
            p.extra_generation_params["SVE Enable"] = False
            p.extra_generation_params[PREFIX+" status"] = "Prepared; no active evaluations yet"
            report(state.summary + "\nWindow: " + state.options["window"] + "; sigma-based denoising progress.", state.options["log"])
        except Exception as exc:
            skip(p, state, str(exc))

    def post_sample(self, p, ps, *values):
        state = getattr(p, ATTR, None)
        if state is not None and state.options is not None and hasattr(state, "summary"):
            report(state.summary + f"\nApplied to {state.calls} model evaluations (not counted steps). Main pass only.", state.options["log"])
            if state.calls == 0:
                p.extra_generation_params[PREFIX+" status"] = "No evaluations in selected window"
            state.active, state.schedules = False, []

    def postprocess(self, p, processed, *values):
        state = getattr(p, ATTR, None)
        if state is not None:
            state.active, state.schedules = False, []


script_callbacks.on_cfg_denoiser(on_cfg)
