import importlib.util
import os
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from krea_sve_core import (DEFAULTS, KEYS, LABELS, RESTORERS, conditioning_noise,
                           validate_options, window_active, stack_noise)


def options(**kwargs):
    return validate_options(dict(DEFAULTS, **kwargs))


def embedding(dtype=torch.float32):
    gen = torch.Generator().manual_seed(16)
    x = torch.randn((1, 9, 128), generator=gen, dtype=dtype)
    x[:, 3] = 0  # Interior null row.
    x[:, 7:] = 0  # Trailing padding.
    return x


class CoreTests(unittest.TestCase):
    def test_determinism_and_sampler_rng_isolation(self):
        x = embedding()
        original = x.clone()
        torch.manual_seed(567)
        rng = torch.get_rng_state().clone()
        a, _ = conditioning_noise(x, options(), 7)
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        b, _ = conditioning_noise(x, options(), 7)
        c, _ = conditioning_noise(x, options(), 8)
        self.assertTrue(torch.equal(a, b))
        self.assertFalse(torch.equal(a, c))
        self.assertTrue(torch.equal(x, original))

    def test_auto_strength_calibrates_to_real_fp32_embeddings(self):
        x = embedding(torch.bfloat16)
        _, diag = conditioning_noise(x, options(auto_strength_factor=2), 5)
        self.assertEqual(diag.std, x[:, :7].float().std().item())
        self.assertEqual(diag.strength, diag.std * 2)
        self.assertEqual(diag.tokens, 7)
        self.assertGreater(diag.l2_percent, 0)

    def test_protects_prefix_suffix_interior_and_trailing_nulls(self):
        for side in ("beginning", "end"):
            noise, _ = conditioning_noise(embedding(), options(protect_from=side, protect_percent=50, randomize_percent=100), 55)
            self.assertEqual(torch.count_nonzero(noise[:, 3]).item(), 0)
            self.assertEqual(torch.count_nonzero(noise[:, 7:]).item(), 0)
            protected = noise[:, :3] if side == "beginning" else noise[:, 4:]
            self.assertEqual(torch.count_nonzero(protected).item(), 0)
            self.assertGreater(torch.count_nonzero(noise).item(), 0)

    def test_token_mode_keeps_all_layers_together(self):
        x = torch.ones((1, 40, 12*2560))
        delta, _ = conditioning_noise(x, options(auto_strength_factor=0, strength=1, granularity="tokens"), 12)
        counts = delta.reshape(40, 12, 2560).ne(0).sum((1, 2))
        self.assertTrue(((counts == 0) | (counts == 12*2560)).all())
        self.assertTrue((counts == 0).any())
        self.assertTrue((counts > 0).any())

    def test_all_padding_and_single_value_are_finite(self):
        for shape in ((1, 3, 8), (1, 1, 1)):
            noise, diag = conditioning_noise(torch.zeros(shape), options(), 0)
            self.assertTrue(torch.isfinite(noise).all())
            self.assertEqual(noise.count_nonzero(), 0)
            self.assertEqual(diag.l2_percent, 0)
            self.assertEqual(diag.std, 0)

    def test_manual_zero_is_identity(self):
        noise, _ = conditioning_noise(embedding(), options(auto_strength_factor=0, strength=0), 5)
        self.assertEqual(noise.count_nonzero(), 0)

    def test_seed_max_preserves_64_bits_and_wraps_mask_seed(self):
        seed = str(2**64-1)
        validated = options(seed=seed)
        noise, diag = conditioning_noise(embedding(), validated, int(seed))
        self.assertEqual(diag.seed, int(seed))
        self.assertTrue(torch.isfinite(noise).all())
        self.assertEqual(RESTORERS["seed"]({LABELS["seed"]: seed}), seed)

    def test_validation_rejects_nonfinite_and_invalid_options(self):
        for kwargs in ({"auto_strength_factor": float("nan")}, {"strength": float("inf")},
                       {"seed": "1.5"}, {"seed": str(2**64)}, {"seed": "-1"},
                       {"window": "Unknown"}, {"randomize_percent": 0}):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                options(**kwargs)

    def test_bad_conditioning_rejected(self):
        for x in (torch.zeros(1, 2), torch.zeros(1, 0, 3), torch.full((1, 2, 3), float("nan"))):
            with self.assertRaises(ValueError):
                conditioning_noise(x, options(), 0)

    def test_sigma_window_does_not_depend_on_evaluation_counts(self):
        sigmas = [1, .9, .85, .7, .69, .5, .3, .2, .01]
        self.assertEqual([window_active("Beginning", s, .7) for s in sigmas], [True]*3+[False]*6)
        self.assertEqual([window_active("Ending", s, .7) for s in sigmas], [False]*3+[True]*6)
        self.assertTrue(all(window_active("All", s, .7) for s in sigmas))

    def test_stack_does_not_perturb_batch_padding_or_original(self):
        rows = [torch.ones(2, 12, 8), torch.ones(4, 12, 8)*2]
        target = torch.ones(2, 4, 12, 8)*10
        noise = stack_noise(rows, target)
        self.assertTrue(torch.equal(noise[0, :2], rows[0]))
        self.assertEqual(noise[0, 2:].count_nonzero(), 0)
        self.assertTrue(torch.equal(noise[1], rows[1]))
        self.assertTrue((target == 10).all())
        with self.assertRaises(ValueError):
            stack_noise(rows, target[0])

    def test_metadata_restore_defaults_and_random_replay(self):
        self.assertFalse(RESTORERS["enabled"]({}))
        self.assertFalse(RESTORERS["enabled"]({LABELS["enabled"]: "False"}))
        params = {LABELS[k]: v for k, v in options(enabled=True, seed_mode="Random per image", seed="123456789").items()}
        self.assertEqual(RESTORERS["seed_mode"](params), "Fixed")
        restored = validate_options({k: RESTORERS[k](params) for k in KEYS})
        self.assertTrue(restored["enabled"])
        self.assertEqual(restored["seed"], "123456789")


@unittest.skipUnless(os.environ.get("KREA_SVE_UPSTREAM"), "Set KREA_SVE_UPSTREAM to the pinned original source for parity checks")
class UpstreamParityTests(unittest.TestCase):
    def test_exact_original_tensor_math_across_modes_and_dtypes(self):
        def set_values(conditioning, values):
            return [[t, dict(meta, **values)] for t, meta in conditioning]
        helper = types.SimpleNamespace(conditioning_set_values=set_values)
        with patch.dict(sys.modules, {"node_helpers": helper}):
            spec = importlib.util.spec_from_file_location("krea_sve_reference", os.environ["KREA_SVE_UPSTREAM"])
            upstream = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(upstream)
        original = upstream.KreaSeedVarianceEnhancer()
        # CPU only: do not initialize CUDA alongside a live Forge generation.
        with patch.object(torch.cuda, "is_available", return_value=False):
            for dtype in (torch.float32, torch.bfloat16, torch.float16):
                for distribution in ("uniform", "gaussian"):
                    for granularity in ("values", "tokens"):
                        for side in ("beginning", "end"):
                            for auto, strength in ((1., 20.), (0., 2.), (0., 1000000002.)):
                                with self.subTest(dtype=dtype, distribution=distribution, granularity=granularity, side=side, auto=auto, strength=strength):
                                    emb = embedding(dtype)
                                    o = options(distribution=distribution, granularity=granularity, protect_from=side,
                                                protect_percent=28, auto_strength_factor=auto, strength=strength)
                                    delta, _ = conditioning_noise(emb, o, 2**64-1)
                                    ref, _ = original.randomize_conditioning([[emb, {}]], 50, auto, strength,
                                        "noise on all steps", 25, 2**64-1, side, 28, False, distribution, granularity)
                                    self.assertTrue(torch.equal(emb+delta, ref[0][0]))


if __name__ == "__main__":
    unittest.main()
