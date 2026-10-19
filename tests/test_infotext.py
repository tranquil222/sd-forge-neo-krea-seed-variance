"""Exercise the installed Forge formatter/parser without booting Forge/GPU."""
import ast
import io
import json
import os
from pathlib import Path
import re
import sys
from types import SimpleNamespace as NS
import unittest
from unittest.mock import patch

import gradio as gr
from PIL import Image, PngImagePlugin

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_adapter import adapter, processing, conditioning, tensor
from krea_sve_core import KEYS, DEFAULTS, LABELS, validate_options

FORGE = Path(os.environ.get("FORGE_ROOT", Path(__file__).resolve().parents[3]))


class Defaults(NS):
    def __getattr__(self, name):
        return False


def installed_functions(path, names, namespace):
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.Assign)) and
             (getattr(node, "name", None) in names or
              any(isinstance(t, ast.Name) and t.id in names for t in getattr(node, "targets", [])))]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@unittest.skipUnless((FORGE / "modules/infotext_utils.py").is_file(), "Set FORGE_ROOT for host PNG tests")
class InfotextTests(unittest.TestCase):
    def test_installed_forge_formatter_png_parser_and_component_restore(self):
        opts = Defaults(forge_additional_modules=[], sd_model_checkpoint="test.safetensors",
                        strip_whitespaces=True, infotext_skip_pasting=[], forge_unet_storage_dtype="Automatic")
        parser = installed_functions(FORGE/"modules/infotext_utils.py",
            {"quote", "unquote", "re_param_code", "re_param", "re_imagesize", "parse_generation_parameters", "_parse_info"},
            dict(json=json, re=re, os=os, Any=object, gr=gr, type_of_gr_update=type(gr.skip()), shared=NS(opts=opts),
                 _extract_styles=lambda res, prompt, neg: (prompt, neg), _populate_defaults=lambda res: None,
                 uses_emphasis=lambda text: False, main_entry=NS(module_list={}),
                 errors=NS(report=lambda *a, **k: self.fail(str(a)))))
        formatter = installed_functions(FORGE/"modules/processing.py", {"create_infotext"},
            dict(opts=opts, shared=NS(opts=opts), _overridden_modules=None,
                 infotext_utils=NS(quote=parser["quote"]),
                 inpaint_metadata=NS(infotext=lambda p, current, *args: current),
                 errors=NS(report=lambda *a, **k: self.fail(str(a)))))
        p = Defaults(**vars(processing((10, 10, 10, 10), batch_size=2)))
        p.refiner_checkpoint_info = None
        p.width = p.height = 1024
        p.steps, p.sampler_name, p.scheduler, p.cfg_scale = 15, "RES4LYF res_2m", "Simple", 1
        p.all_prompts, p.all_negative_prompts = ["A landscape"]*4, [""]*4
        p.main_prompt = p.all_prompts[0]
        p.get_token_merging_ratio = lambda **kwargs: 0
        p.sd_model.use_distilled_cfg_scale = p.sd_model.use_shift = p.sd_model.is_sd1 = False
        p.sd_model.model_config = NS(model_type=NS(name="CONST"))
        script = adapter.KreaSeedVariance()
        selected = validate_options(dict(DEFAULTS, enabled=True, log=False, seed_mode="Random per image",
                                         distribution="gaussian", granularity="tokens", protect_percent=20,
                                         window="All", auto_strength_factor=.65, switch_percent=45))
        with patch.object(adapter.secrets, "randbits", side_effect=[101, 102, 103, 2**64-1]):
            script.process(p, *(selected[k] for k in KEYS))
        with gr.Blocks(analytics_enabled=False):
            controls = script.ui(False)
        for batch in (0, 1):
            p.iteration, p.seeds = batch, p.all_seeds[batch*2:batch*2+2]
            script.process_before_every_sampling(p, c=conditioning(tensor(), tensor()))
            for i in (0, 1):
                text = formatter["create_infotext"](p, p.all_prompts[:2], p.seeds, [0, 0], index=i)
                info = PngImagePlugin.PngInfo()
                info.add_text("parameters", text)
                file = io.BytesIO()
                Image.new("RGB", (2, 2)).save(file, format="PNG", pnginfo=info)
                file.seek(0)
                parsed = parser["parse_generation_parameters"](Image.open(file).info["parameters"])
                self.assertEqual(parsed["SVE Enable"], "False")
                self.assertEqual(parsed["Krea SVE seed"], str([101, 102, 103, 2**64-1][batch*2+i]))
                restored = [parser["_parse_info"](component, fn, parsed)["value"]
                            for component, fn in script.infotext_fields]
                actual = validate_options(dict(zip(KEYS, restored)))
                expected = dict(selected, seed=parsed["Krea SVE seed"], seed_mode="Fixed")
                self.assertEqual(actual, expected)
        # Result/gallery replay uses the global index, not the current batch.
        text = formatter["create_infotext"](p, p.all_prompts, p.all_seeds, [0]*4, index=0)
        self.assertEqual(parser["parse_generation_parameters"](text)["Krea SVE seed"], "101")


if __name__ == "__main__": unittest.main()
