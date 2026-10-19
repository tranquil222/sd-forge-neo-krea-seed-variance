from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import sys
import unittest
from types import SimpleNamespace as NS
from unittest.mock import patch

import gradio as gr
import torch
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_fixture import load_adapter
from krea_sve_core import DEFAULTS, KEYS, LABELS, RESTORERS, TITLE, conditioning_noise, validate_options

adapter, callbacks = load_adapter()


def tensor(tokens=4):
    return torch.randn(tokens, 12, 2560, generator=torch.Generator().manual_seed(33))


def conditioning(*embeddings):
    return NS(batch=[[NS(schedules=[NS(cond=emb, end_at_step=1000)], weight=1.)] for emb in embeddings])


def processing(seeds=(10,), batch_size=1, **extra):
    model_type = type("Krea2", (), {"__module__": "backend.diffusion_engine.krea"})
    model = model_type()
    model.forge_objects = NS(unet=NS(model=NS(predictor=NS(percent_to_sigma=lambda percent: 1-percent))))
    return NS(sd_model=model, all_seeds=list(seeds), seeds=list(seeds[:batch_size]), batch_size=batch_size,
              iteration=0, is_hr_pass=False, extra_generation_params={}, **extra)


def params(p, cond, sigma=.95, step=0):
    return NS(denoiser=NS(p=p, step=step), text_cond=cond.clone(), text_uncond=torch.ones_like(cond),
              sigma=torch.tensor([sigma]), sampling_step=99, total_sampling_steps=15)


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.script = adapter.KreaSeedVariance()
        self.stdout = redirect_stdout(io.StringIO())
        self.stdout.__enter__()
        self.warning = patch.object(adapter.gr, "Warning")
        self.warning.__enter__()

    def tearDown(self):
        self.warning.__exit__(None, None, None)
        self.stdout.__exit__(None, None, None)

    def start(self, p, cond, **options):
        options = dict(DEFAULTS, enabled=True, log=False, **options)
        self.script.process(p, *(options[k] for k in KEYS))
        self.script.process_before_every_sampling(p, c=cond)
        return getattr(p, adapter.ATTR)

    def test_real_gradio_panel_and_callback_registration(self):
        with gr.Blocks() as ui:
            controls = self.script.ui(False)
        self.assertEqual(len(controls), len(KEYS))
        self.assertEqual([c.elem_id for c in controls], ["krea_sve_txt2img_"+k for k in KEYS])
        self.assertEqual([c.value for c in controls], list(DEFAULTS.values()))
        self.assertEqual(len(self.script.infotext_fields), len(KEYS))
        self.assertEqual(callbacks, [adapter.on_cfg])
        self.assertFalse(self.script.show(True))
        self.assertEqual(self.script.title(), TITLE)
        json.dumps(ui.get_config_file())

    def test_callback_matches_original_flattened_math_and_keeps_negative_and_cache(self):
        emb = tensor()
        original = emb.clone()
        p = processing()
        state = self.start(p, conditioning(emb))
        self.assertTrue(state.active)
        call = params(p, emb.unsqueeze(0))
        neg = call.text_uncond.clone()
        expected, _ = conditioning_noise(emb.reshape(1, 4, -1), validate_options(DEFAULTS), 10)
        adapter.on_cfg(call)
        self.assertTrue(torch.equal(call.text_cond, original.unsqueeze(0)+expected.reshape(1, 4, 12, 2560)))
        self.assertTrue(torch.equal(emb, original))
        self.assertTrue(torch.equal(call.text_uncond, neg))
        self.assertEqual(state.calls, 1)
        self.assertEqual(p.extra_generation_params["Krea SVE status"], "Applied (main pass)")

    def test_beginning_stops_and_all_persists_through_substeps(self):
        emb = tensor()
        for window, expected_calls in (("Beginning", 2), ("Ending", 3), ("All", 5)):
            p = processing()
            state = self.start(p, conditioning(emb), window=window)
            for step, sigma in enumerate((1., .8, .75, .5, .01)):
                call = params(p, emb.unsqueeze(0), sigma, step*15)
                adapter.on_cfg(call)
            self.assertEqual(state.calls, expected_calls)

    def test_batch_seed_independence_prompt_schedules_branches_and_padding(self):
        short, long = tensor(2), tensor(5)
        c = conditioning(short, long)
        c.batch[0][0].schedules.append(NS(cond=long, end_at_step=1000))
        c.batch[0][0].schedules[0].end_at_step = 1
        c.batch[1].append(NS(schedules=[NS(cond=long, end_at_step=1000)], weight=.5))
        p = processing((10, 11), batch_size=2)
        state = self.start(p, c, window="All")
        self.assertEqual(len(state.schedules), 3)
        target = torch.zeros(3, 5, 12, 2560)
        first = params(p, target, step=0)
        adapter.on_cfg(first)
        self.assertEqual(first.text_cond[0, 2:].count_nonzero(), 0)
        late = params(p, target, step=2)
        adapter.on_cfg(late)
        self.assertGreater(late.text_cond[0, 2:].count_nonzero(), 0)
        self.assertFalse(torch.equal(late.text_cond[0], late.text_cond[1]))
        self.assertTrue(torch.equal(late.text_cond[1], late.text_cond[2]))

    def test_same_noise_reused_and_composes_with_other_positive_changes(self):
        emb = tensor()
        p = processing()
        state = self.start(p, conditioning(emb), window="All")
        ptr = state.schedules[0][0].cond.data_ptr()
        a, b = params(p, emb.unsqueeze(0)), params(p, emb.unsqueeze(0)+1, step=50)
        adapter.on_cfg(a)
        adapter.on_cfg(b)
        torch.testing.assert_close(b.text_cond, a.text_cond+1)
        self.assertEqual(state.schedules[0][0].cond.data_ptr(), ptr)

    def test_disabled_is_exact_identity_and_old_metadata_removed(self):
        p, emb = processing(), tensor()
        p.extra_generation_params.update({"Krea SVE": True, "Other": 5})
        self.script.process(p, *(DEFAULTS[k] for k in KEYS))
        self.script.process_before_every_sampling(p, c=conditioning(emb))
        call = params(p, emb.unsqueeze(0))
        before = call.text_cond
        adapter.on_cfg(call)
        self.assertIs(call.text_cond, before)
        self.assertEqual(p.extra_generation_params, {"Other": 5})

    def test_legacy_conflict_skips_new_without_changing_original(self):
        p = processing()
        p.extra_generation_params["SVE Enable"] = True
        state = self.start(p, conditioning(tensor()))
        self.assertFalse(state.active)
        self.assertTrue(p.extra_generation_params["SVE Enable"])
        self.assertIn("original Seed Variance", p.extra_generation_params["Krea SVE status"])

    def test_wrong_model_refiner_and_layout_report_explicit_skip(self):
        for fault in ("model", "refiner", "layout"):
            p, emb = processing(), tensor()
            if fault == "model": p.sd_model = NS()
            if fault == "refiner": p.refiner_checkpoint_info = object()
            if fault == "layout": emb = emb[:, :2]
            state = self.start(p, conditioning(emb))
            self.assertFalse(state.active)
            self.assertTrue(p.extra_generation_params["Krea SVE status"].startswith("Skipped:"))

    def test_hires_is_untouched_and_main_diagnostics_retained(self):
        emb, p = tensor(), processing()
        state = self.start(p, conditioning(emb), window="All")
        adapter.on_cfg(params(p, emb.unsqueeze(0)))
        p.is_hr_pass = True
        self.script.process_before_every_sampling(p, c=conditioning(emb))
        call = params(p, emb.unsqueeze(0))
        before = call.text_cond
        adapter.on_cfg(call)
        self.assertIs(call.text_cond, before)
        self.script.post_sample(p, None)
        self.assertIn("Applied to 1 model evaluations", adapter.diagnostics_text())
        self.assertEqual(state.schedules, [])

    def test_random_actual_seeds_saved_per_image_across_batches_and_png_replay(self):
        p = processing((10, 10, 10, 10), batch_size=2)
        with patch.object(adapter.secrets, "randbits", side_effect=[101, 102, 103, 2**64-1]):
            self.start(p, conditioning(tensor(), tensor()), seed_mode="Random per image")
        getter = p.extra_generation_params[LABELS["seed"]]
        self.assertEqual(getter(p, 1, p.seeds), "102")
        p.iteration = 1
        p.seeds = p.all_seeds[2:]
        self.assertEqual(getter(p, 0, p.seeds), "103")
        self.assertEqual(getter(p, 3, p.all_seeds), str(2**64-1))
        params_saved = dict(p.extra_generation_params)
        params_saved[LABELS["seed"]] = getter(p, 1, p.seeds)
        info = PngImagePlugin.PngInfo()
        info.add_text("parameters", json.dumps(params_saved))
        file = io.BytesIO()
        Image.new("RGB", (2, 2)).save(file, format="PNG", pnginfo=info)
        file.seek(0)
        loaded = json.loads(Image.open(file).info["parameters"])
        restored = validate_options({k: RESTORERS[k](loaded) for k in KEYS})
        self.assertEqual(restored["seed_mode"], "Fixed")
        self.assertEqual(restored["seed"], str(2**64-1))
        before, _ = conditioning_noise(tensor().reshape(1, 4, -1), restored, 2**64-1)
        after, _ = conditioning_noise(tensor().reshape(1, 4, -1), restored, int(restored["seed"]))
        self.assertTrue(torch.equal(before, after))

    def test_invalid_options_cannot_apply_partially(self):
        p = processing()
        state = self.start(p, conditioning(tensor()), auto_strength_factor=float("nan"))
        self.assertFalse(state.active)
        self.assertIsNone(state.options)

    def test_follow_mode_tracks_actual_seeds_finalized_by_other_scripts(self):
        p, emb = processing(), tensor()
        options = dict(DEFAULTS, enabled=True, log=False)
        self.script.process(p, *(options[k] for k in KEYS))
        p.seeds = [991]
        self.script.process_before_every_sampling(p, c=conditioning(emb))
        state = getattr(p, adapter.ATTR)
        expected, _ = conditioning_noise(emb.reshape(1, 4, -1), validate_options(options), 991)
        self.assertTrue(torch.equal(state.schedules[0][0].cond, expected.reshape_as(emb)))
        self.assertEqual(adapter.metadata_seed(p, 0, p.seeds), "991")

    def test_no_window_evaluations_are_reported_instead_of_claiming_applied(self):
        p, emb = processing(), tensor()
        self.start(p, conditioning(emb))
        adapter.on_cfg(params(p, emb.unsqueeze(0), sigma=.01))
        self.script.post_sample(p, None)
        self.assertEqual(p.extra_generation_params["Krea SVE status"], "No evaluations in selected window")


if __name__ == "__main__": unittest.main()
