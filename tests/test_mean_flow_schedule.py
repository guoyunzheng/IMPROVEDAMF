"""CPU checks for the epoch schedule, trajectory sampling, and FM/MF gradients."""

import ast
from functools import partial
from pathlib import Path
from types import MethodType, SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
import torch

from source.common.mean_flow_schedule import scheduled_flow_ratio


SCHEDULE = {"enabled": True, "fm_only_fraction": 0.8}
POLICY_FILES = (
    "mean_flow_transformer_lowdim_policy.py",
    "adaptive_mean_flow_transformer_lowdim_policy.py",
    "mean_flow_transformer_image_policy.py",
    "adaptive_mean_flow_transformer_image_policy.py",
)


class ScheduleTests(unittest.TestCase):
    def test_first_400_of_500_epochs_are_pure_fm(self):
        ratios = [scheduled_flow_ratio(epoch, 500, 0.2, SCHEDULE)
                  for epoch in range(500)]
        self.assertEqual(ratios, [1.0] * 400 + [0.2] * 100)

    def test_fractional_boundary_rounds_up_to_keep_first_80_percent_pure_fm(self):
        self.assertEqual(scheduled_flow_ratio(400, 501, 0.2, SCHEDULE), 1.0)
        self.assertEqual(scheduled_flow_ratio(401, 501, 0.2, SCHEDULE), 0.2)
        self.assertEqual(scheduled_flow_ratio(500, 501, 0.2, SCHEDULE), 0.2)

    def test_resuming_in_mixed_phase_does_not_restart_pure_fm(self):
        remaining = [scheduled_flow_ratio(epoch, 500, 0.2, SCHEDULE)
                     for epoch in range(450, 500)]
        self.assertEqual(remaining, [0.2] * 50)

    def test_missing_or_disabled_schedule_keeps_legacy_fixed_ratio(self):
        for schedule in (None, {"enabled": False}):
            for epoch in (0, 400, 499):
                self.assertEqual(scheduled_flow_ratio(epoch, 500, 0.5, schedule), 0.5)

    def test_zero_and_full_pure_fm_phase(self):
        self.assertEqual(scheduled_flow_ratio(
            0, 500, 0.2, {"fm_only_fraction": 0.0}), 0.2)
        self.assertEqual(scheduled_flow_ratio(
            499, 500, 0.2, {"fm_only_fraction": 1.0}), 1.0)

    def test_invalid_progress_and_ratios_are_rejected(self):
        for epoch, total, ratio, schedule in (
            (-1, 500, 0.2, SCHEDULE), (0, 0, 0.2, SCHEDULE),
            (0, 500, -0.1, SCHEDULE), (0, 500, 1.1, SCHEDULE),
            (0, 500, 0.2, {"fm_only_fraction": -0.1}),
            (0, 500, 0.2, {"fm_only_fraction": 1.1}),
        ):
            with self.subTest(epoch=epoch, total=total, ratio=ratio, schedule=schedule):
                with self.assertRaises(ValueError):
                    scheduled_flow_ratio(epoch, total, ratio, schedule)


class IdentityNormalizer:
    def normalize(self, value):
        return value

    def __getitem__(self, key):
        return self


class TinyVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(0.4))

    def forward(self, z, t, r, cond=None):
        t = t.reshape(t.shape[0], -1, 1)
        r = r.reshape(r.shape[0], -1, 1)
        return self.scale * (z + 0.3 * t + 0.2 * r)


def training_fixture(filename, flow_ratio):
    # Load the actual sampling/loss methods without importing optional vision,
    # dataset, or simulation packages. Only the model and normalization are fakes.
    path = Path(__file__).resolve().parents[1] / "source/policy" / filename
    tree = ast.parse(path.read_text(encoding="utf-8"))
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
    functions = [node for node in tree.body
                 if isinstance(node, ast.FunctionDef) and node.name == "adaptive_l2_loss"]
    functions.extend(node for node in cls.body if isinstance(node, ast.FunctionDef)
                     and node.name in {"sample_t_r", "compute_loss"})
    namespace = {
        "np": np, "torch": torch, "partial": partial,
        # Both production rearrange patterns just append singleton dimensions.
        "rearrange": lambda tensor, pattern: tensor.reshape(tensor.shape[0], -1, 1),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), namespace)
    policy = SimpleNamespace(
        flow_ratio=flow_ratio, horizon=6, n_obs_steps=2,
        monotone_schedule_ratio=0.5, time_dist=["lognorm", -0.4, 1.0],
        normalizer=IdentityNormalizer(), model=TinyVelocity(),
    )
    policy.sample_t_r = MethodType(namespace["sample_t_r"], policy)
    policy.compute_loss = MethodType(namespace["compute_loss"], policy)
    policy.model_call = lambda z, t, r, cond: policy.model(z, t, r, cond=cond)
    policy.process_obs = lambda obs: obs[:, :2].reshape(-1, obs.shape[-1])
    return policy


class SamplingAndLossTests(unittest.TestCase):
    def setUp(self):
        np.random.seed(42)
        torch.manual_seed(42)
        self.batch = {"obs": torch.randn(10, 6, 3), "action": torch.randn(10, 6, 2)}

    def test_epoch_switch_applies_to_entire_trajectories_in_all_four_policies(self):
        for filename in POLICY_FILES:
            for epoch, fm_count in ((399, 10), (400, 2)):
                with self.subTest(policy=filename, epoch=epoch):
                    ratio = scheduled_flow_ratio(epoch, 500, 0.2, SCHEDULE)
                    policy = training_fixture(filename, ratio)
                    t, r = policy.sample_t_r(10, "cpu")
                    equal = (t == r).reshape(10, -1)
                    self.assertEqual(int(equal.all(dim=1).sum()), fm_count)
                    self.assertTrue(torch.equal(equal.any(dim=1), equal.all(dim=1)))
                    self.assertTrue((r <= t).all())

    def test_pure_fm_skips_jvp_and_matches_degenerate_mf_loss_and_gradient(self):
        for filename in POLICY_FILES:
            with self.subTest(policy=filename):
                policy = training_fixture(filename, 1.0)
                t, r = policy.sample_t_r(10, "cpu")
                policy.sample_t_r = lambda batch_size, device: (t, r)
                noise = torch.randn_like(self.batch["action"])
                with patch.object(torch, "randn_like", return_value=noise):
                    with patch.object(torch.autograd.functional, "jvp",
                                      side_effect=AssertionError("Pure FM must skip JVP")):
                        fm_loss = policy.compute_loss(self.batch)
                    fm_gradient = torch.autograd.grad(fm_loss, policy.model.scale)[0]

                    # Exercise the original JVP loss with the same r=t samples.
                    policy.flow_ratio = 0.2
                    mf_loss = policy.compute_loss(self.batch)
                    mf_gradient = torch.autograd.grad(mf_loss, policy.model.scale)[0]
                torch.testing.assert_close(fm_loss, mf_loss)
                torch.testing.assert_close(fm_gradient, mf_gradient)

    def test_mixed_phase_uses_jvp_and_has_finite_gradients(self):
        for filename in POLICY_FILES:
            with self.subTest(policy=filename):
                policy = training_fixture(filename, 0.2)
                with patch.object(torch.autograd.functional, "jvp",
                                  wraps=torch.autograd.functional.jvp) as jvp:
                    loss = policy.compute_loss(self.batch)
                jvp.assert_called_once()
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(policy.model.scale.grad))


if __name__ == "__main__":
    unittest.main()
