"""UI recipes only: generation/PNG/queue values remain the existing controls."""
from krea_sve_core import DEFAULTS

CUSTOM = "Custom / current settings"
# Never include enabled, either seed control, or logging in a recipe.
PRESET_KEYS = ("auto_strength_factor", "randomize_percent", "window", "switch_percent",
               "distribution", "granularity", "protect_from", "protect_percent", "strength")


def _recipe(description, **changes):
    values = {key: DEFAULTS[key] for key in PRESET_KEYS}
    values.update(window="All", distribution="gaussian", granularity="tokens")
    values.update(changes)
    return dict(description=description, values=values)


PRESETS = {
    "Light": _recipe("Gentle whole-token variation throughout sampling.",
                     auto_strength_factor=0.5, randomize_percent=30.0),
    "Light+": _recipe("A stronger Light setting, with the same 30% whole-token selection throughout sampling.",
                      auto_strength_factor=0.75, randomize_percent=30.0),
    "Medium": _recipe("Balanced starting point: whole-token variation throughout sampling.",
                      auto_strength_factor=1.0, randomize_percent=50.0),
    "Medium+": _recipe("Medium with auto strength raised to 1.5; keeps the same 50% whole-token selection throughout sampling.",
                       auto_strength_factor=1.5, randomize_percent=50.0),
    "Heavy": _recipe("Stronger whole-token variation throughout sampling; more prompt and identity drift is possible.",
                     auto_strength_factor=2.0, randomize_percent=65.0),
    "Heavy+": _recipe("Higher-strength Heavy variation, with the same 65% token selection. Prompt and identity drift can increase.",
                      auto_strength_factor=3.0, randomize_percent=65.0),
    "Chaos": _recipe("Experimental, high-strength variation throughout sampling. Expect prompt drift and possible artifacts.",
                     auto_strength_factor=4.0, randomize_percent=80.0),
    "Chaos+": _recipe("Very high-strength experimental variation. Keeps Chaos's 80% token selection; artifacts and major prompt drift are possible.",
                      auto_strength_factor=6.0, randomize_percent=80.0),
    "Early exploration": _recipe("Noisy conditioning early, then the clean prompt returns. The final image may settle back toward the usual composition.",
                                 window="Beginning", switch_percent=40.0),
    "Late variation": _recipe("Leaves early conditioning unchanged, then adds light feature-level noise late. Not a composition lock or a guaranteed texture improvement.",
                              auto_strength_factor=0.5, randomize_percent=30.0,
                              window="Ending", switch_percent=65.0, granularity="values"),
    "Protect prompt opening": _recipe("Medium variation while leaving the first 40% of encoded prompt tokens untouched. This does not isolate or protect a particular LoRA.",
                                      protect_from="beginning", protect_percent=40.0),
    "Original defaults": dict(description="The original node's starting recipe: uniform, value-level noise early, then clean conditioning.",
                              values={key: DEFAULTS[key] for key in PRESET_KEYS}),
}
CHOICES = (CUSTOM, *PRESETS)


def preset_values(name):
    """Return a fresh mapping; Custom/unknown labels never change controls."""
    recipe = PRESETS.get(name)
    return dict(recipe["values"]) if recipe is not None else None


def matching_preset(values):
    for name, recipe in PRESETS.items():
        for key, expected in recipe["values"].items():
            actual = values.get(key)
            # Infotext/API restores may supply numeric strings.
            if isinstance(expected, (int, float)):
                try:
                    same = abs(float(actual)-expected) < 1e-9
                except (TypeError, ValueError):
                    same = False
            else:
                same = actual == expected
            if not same:
                break
        else:
            return name
    return CUSTOM


def preset_description(name):
    recipe = PRESETS.get(name)
    if recipe is None:
        return "Custom settings. Choose a preset to fill the effect controls, or adjust them below."
    values = recipe["values"]
    window = "all sampling" if values["window"] == "All" else (
        f"before {values['switch_percent']:g}% progress" if values["window"] == "Beginning" else
        f"after {values['switch_percent']:g}% progress")
    return (recipe["description"] + f"  \n**{values['auto_strength_factor']:g}× auto strength · "
            f"{values['randomize_percent']:g}% {values['granularity']} · {window}**")
