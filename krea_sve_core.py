"""Krea SVE tensor port. Adapted from HARUKI's MIT-0 node; see UPSTREAM.md."""
from dataclasses import dataclass
import math

import torch

VERSION = "0.1.2"
TITLE = "Krea Seed Variance Enhancer"
PREFIX = "Krea SVE"
WINDOWS = ("Beginning", "Ending", "All")
SEED_MODES = ("Follow image seed", "Fixed", "Random per image")
KEYS = ("enabled", "randomize_percent", "auto_strength_factor", "strength", "window",
        "switch_percent", "seed_mode", "seed", "protect_from", "protect_percent",
        "log", "distribution", "granularity")
DEFAULTS = dict(zip(KEYS, (False, 50.0, 1.0, 20.0, "Beginning", 25.0,
                          "Follow image seed", "0", "beginning", 0.0, True, "uniform", "values")))
LABELS = dict(zip(KEYS, (PREFIX, PREFIX+" percent", PREFIX+" auto strength", PREFIX+" manual strength",
                        PREFIX+" window", PREFIX+" switch percent", PREFIX+" seed mode", PREFIX+" seed",
                        PREFIX+" protect from", PREFIX+" protect percent", PREFIX+" log",
                        PREFIX+" distribution", PREFIX+" granularity")))


def as_bool(value):
    return value is True or str(value).strip().lower() in ("true", "1")


def validate_options(values):
    out = dict(DEFAULTS, **values)
    for key in ("enabled", "log"):
        out[key] = as_bool(out[key])
    for key, low, high in (("randomize_percent", 1, 100), ("auto_strength_factor", 0, 100),
                           ("strength", -4294967295, 4294967295), ("switch_percent", 1, 99),
                           ("protect_percent", 0, 99)):
        number = float(out[key])
        if not math.isfinite(number) or not low <= number <= high:
            raise ValueError(f"{key} must be between {low} and {high}")
        out[key] = number
    for key, choices in (("window", WINDOWS), ("seed_mode", SEED_MODES),
                         ("protect_from", ("beginning", "end")),
                         ("distribution", ("uniform", "gaussian")), ("granularity", ("values", "tokens"))):
        if out[key] not in choices:
            raise ValueError(f"Invalid {key}: {out[key]}")
    seed = str(out["seed"]).strip()
    if not seed.isascii() or not seed.isdecimal() or not 0 <= int(seed) < (1 << 64):
        raise ValueError("Variation seed must be a whole number from 0 to 18446744073709551615")
    out["seed"] = str(int(seed))  # Text preserves all 64 bits through the browser/PNG.
    return out


def restore_field(key, params):
    value = params.get(LABELS[key], DEFAULTS[key])
    if key in ("enabled", "log"):
        return as_bool(value)
    if key == "seed_mode" and value == "Random per image":
        return "Fixed"  # PNG replay uses the recorded seed, not a fresh random seed.
    if key == "seed":
        return str(value)
    return value


# Shared function identities also work with Forge's combined paste-field lists.
RESTORERS = {key: (lambda params, key=key: restore_field(key, params)) for key in KEYS}


@dataclass(frozen=True)
class Diagnostics:
    seed: int
    tokens: int
    total_tokens: int
    std: float
    strength: float
    selected_percent: float
    l2_percent: float

    def text(self):
        return (f"seed {self.seed} | tokens {self.tokens}/{self.total_tokens} | "
                f"embedding std {self.std:.5g} | effective strength {self.strength:.5g} | "
                f"selected {self.selected_percent:.1f}% | L2 noise {self.l2_percent:.2f}%")


@torch.inference_mode()
def conditioning_noise(embedding, options, seed):
    """Return additive noise, not an in-place edit. Input is [B, tokens, features].

    Math/RNG match the pinned upstream node. A device-local generator avoids
    even temporarily touching the sampler's global CPU/CUDA random streams.
    """
    if not isinstance(embedding, torch.Tensor) or embedding.ndim != 3 or not embedding.is_floating_point():
        raise ValueError("Expected floating [batch, tokens, features] conditioning")
    if not embedding.numel() or not torch.isfinite(embedding).all():
        raise ValueError("Empty or non-finite conditioning")
    emb = embedding.contiguous()
    seed = int(seed) % (1 << 64)
    nonnull = (emb.abs() >= 1e-6).any(dim=-1).any(dim=0)
    indices = torch.nonzero(nonnull).flatten()
    real_tokens = int(indices[-1]) + 1 if indices.numel() else 0
    real = emb[:, :real_tokens].float()
    measured_std = real.std().item() if real.numel() > 1 else 0.0
    scale = measured_std * options["auto_strength_factor"] if options["auto_strength_factor"] > 0 else options["strength"]
    reset_mask_seed = True
    # Retain upstream's legacy manual-strength encoding.
    if options["auto_strength_factor"] <= 0 and int(scale / 1000000000) == 1:
        scale -= 1000000000
        reset_mask_seed = False
    generator = torch.Generator(device=emb.device).manual_seed(seed)
    if options["distribution"] == "gaussian":
        noise = torch.randn(emb.shape, device=emb.device, dtype=emb.dtype, generator=generator) * (scale / math.sqrt(3))
    else:
        noise = torch.rand(emb.shape, device=emb.device, dtype=emb.dtype, generator=generator) * 2 * scale - scale
    if reset_mask_seed:
        generator.manual_seed((seed + 1) % (1 << 64))
    shape = (*emb.shape[:2], 1) if options["granularity"] == "tokens" else emb.shape
    probabilities = torch.ones(shape, device=emb.device, dtype=emb.dtype) * (options["randomize_percent"] / 100)
    selected = torch.bernoulli(probabilities, generator=generator).bool().expand_as(emb)
    allowed = nonnull.clone()
    protected = int(real_tokens * options["protect_percent"] / 100)
    if protected:
        if options["protect_from"] == "beginning":
            allowed[:protected] = False
        else:
            allowed[real_tokens-protected:] = False
    selected = selected & allowed.view(1, -1, 1)
    noise = noise * selected
    if not torch.isfinite(noise).all():
        raise ValueError("Noise overflow: reduce strength")
    base_norm = real.norm().item()
    l2 = noise[:, :real_tokens].float().norm().item() / base_norm * 100 if base_norm else 0.0
    fraction = selected[:, :real_tokens].float().mean().item() * 100 if real_tokens else 0.0
    return noise, Diagnostics(seed, real_tokens, emb.shape[1], measured_std, scale, fraction, l2)


def window_active(window, sigma, switch_sigma):
    """Comfy-style diffusion progress, independent of model-evaluation counts.

    Use only one branch at the exact boundary (the clean branch for Beginning).
    Comfy's two overlapping conditioning entries can average at that boundary.
    """
    if window == "All":
        return True
    return float(sigma) > switch_sigma if window == "Beginning" else float(sigma) <= switch_sigma


def selected_entry(schedule, step):
    # Match Forge prompt_parser's schedule selection, including its fallback.
    return next((entry for entry in schedule if step <= entry.end_at_step), schedule[0])


def stack_noise(rows, target):
    """Do not perturb synthetic padding Forge adds to unequal-length prompts."""
    if target.ndim != 4 or len(rows) != target.shape[0]:
        raise ValueError("Krea conditioning layout changed; expected [batch, tokens, layers, features]")
    result = torch.zeros_like(target)
    for i, row in enumerate(rows):
        if row.ndim != 3 or row.shape[1:] != target.shape[2:] or row.shape[0] > target.shape[1]:
            raise ValueError("Krea conditioning shape changed during sampling")
        result[i, :row.shape[0]] = row.to(device=target.device, dtype=target.dtype)
    return result
