import json
from pathlib import Path
import sys
import unittest

import gradio as gr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from adapter_fixture import load_adapter
from krea_sve_core import DEFAULTS, KEYS, LABELS, RESTORERS, validate_options
from krea_sve_presets import (PRESETS, PRESET_KEYS, CUSTOM, matching_preset,
                              preset_values, preset_description)

adapter, _ = load_adapter()


class PresetTests(unittest.TestCase):
    def test_all_presets_valid_and_do_not_own_enable_seed_or_logging(self):
        protected = dict(enabled=False, seed_mode="Fixed", seed=str(2**64-1), log=False)
        for name in PRESETS:
            patch = preset_values(name)
            actual = validate_options(dict(DEFAULTS, **protected) | patch)
            self.assertEqual({key: actual[key] for key in protected}, protected)
            self.assertEqual(set(patch), set(PRESET_KEYS))
            self.assertEqual(matching_preset(actual), name)

    def test_intensity_ladder_increases_expected_noise_power(self):
        # With matched noise distributions, power scales with strength² × mask fraction.
        powers = []
        for name in ("Light", "Light+", "Medium", "Medium+", "Heavy", "Heavy+", "Chaos", "Chaos+"):
            values = preset_values(name)
            self.assertEqual(values["window"], "All")
            powers.append(values["auto_strength_factor"]**2 * values["randomize_percent"] / 100)
        self.assertTrue(all(a < b for a, b in zip(powers, powers[1:])))

    def test_custom_and_unknown_selection_do_not_change_controls(self):
        for name in (CUSTOM, "Removed preset", None):
            self.assertIsNone(preset_values(name))
            updates = adapter.apply_preset(name)
            self.assertEqual(len(updates), len(PRESET_KEYS))
            self.assertTrue(all("value" not in update for update in updates))

    def test_manual_tweak_changes_match_but_seed_changes_do_not(self):
        values = dict(DEFAULTS) | preset_values("Medium")
        self.assertEqual(matching_preset(dict(values, seed="18", seed_mode="Fixed")), "Medium")
        self.assertEqual(matching_preset(dict(values, auto_strength_factor=1.05)), CUSTOM)
        self.assertEqual(matching_preset(dict(values, auto_strength_factor=float("nan"))), CUSTOM)

    def test_recipes_cannot_be_mutated_through_returned_values(self):
        values = preset_values("Light")
        values["auto_strength_factor"] = 90
        self.assertNotEqual(preset_values("Light")["auto_strength_factor"], 90)

    def test_png_restores_values_even_when_saved_preset_name_disagrees(self):
        values = dict(DEFAULTS, enabled=True, seed_mode="Random per image", seed="847") | preset_values("Heavy")
        png = {LABELS[k]: v for k, v in values.items()}
        png["Krea SVE preset"] = "Light"  # Stale label must not apply the wrong recipe.
        restored = validate_options({key: RESTORERS[key](png) for key in KEYS})
        self.assertEqual(matching_preset(restored), "Heavy")
        self.assertEqual(restored["seed_mode"], "Fixed")
        self.assertEqual(restored["seed"], "847")

    def test_original_defaults_unchanged_and_descriptions_explain_modes(self):
        self.assertEqual(matching_preset(DEFAULTS), "Original defaults")
        self.assertIn("clean", preset_description("Early exploration"))
        self.assertIn("early conditioning unchanged", preset_description("Late variation"))
        self.assertIn("artifacts", preset_description("Chaos"))

    def test_ui_presets_are_not_generation_inputs_and_only_user_input_applies(self):
        with gr.Blocks(analytics_enabled=False) as ui:
            script = adapter.KreaSeedVariance()
            controls = script.ui(False)
        self.assertEqual([c.elem_id for c in controls], ["krea_sve_txt2img_"+key for key in KEYS])
        self.assertEqual([c.value for c in controls], list(DEFAULTS.values()))
        config = ui.get_config_file()
        picker = next(c for c in config["components"] if c["props"].get("elem_id") == "krea_sve_txt2img_preset")
        self.assertNotIn(picker["id"], [c._id for c in controls])
        handlers = [d for d in config["dependencies"] if any(t[0] == picker["id"] for t in d["targets"])]
        self.assertEqual(len(handlers), 1)
        self.assertEqual(handlers[0]["targets"], [(picker["id"], "input")])
        output_ids = [controls[KEYS.index(key)]._id for key in PRESET_KEYS]
        self.assertEqual(handlers[0]["outputs"], output_ids)
        json.dumps(config)


if __name__ == "__main__": unittest.main()
