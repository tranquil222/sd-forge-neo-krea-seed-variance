"""Real Gradio/Torch, minimal Forge objects; never loads a model or real queue."""
import importlib.util
from pathlib import Path
import sys
import types
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_adapter():
    registered = []
    forge = types.ModuleType("modules")
    forge.scripts = types.SimpleNamespace(Script=type("Script", (), {}), AlwaysVisible=object())
    forge.script_callbacks = types.SimpleNamespace(on_cfg_denoiser=registered.append)
    infotext = types.ModuleType("modules.infotext_utils")
    infotext.PasteField = lambda c, fn: (c, fn)
    with patch.dict(sys.modules, {"modules": forge, "modules.infotext_utils": infotext}):
        spec = importlib.util.spec_from_file_location("krea_sve_adapter_fixture", ROOT/"scripts/krea_seed_variance.py")
        adapter = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(adapter)
    return adapter, registered
