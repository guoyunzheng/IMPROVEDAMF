"""Absolute learned-split progress, including warmup, caps and EMA buffers."""

from types import SimpleNamespace
import unittest

import torch

from source.common.kinematic_mean_flow_schedule import set_kmf_progress


def policy(mode='kmf_learnable', ratio=.2):
    generator = torch.nn.Module()
    generator.register_buffer('progress', torch.tensor(0.))
    return SimpleNamespace(mf_target_type=mode, flow_ratio=ratio,
                           kmf_generator=generator, kmf_gamma_max=.5)


class KinematicScheduleTests(unittest.TestCase):
    def test_excludes_warmup_and_uses_actual_capped_batches_with_ema(self):
        model, ema = policy(), policy()
        values = []
        for epoch in range(4):
            model.flow_ratio = 1. if epoch == 0 else .2
            for batch_idx in range(2):
                row = set_kmf_progress(model, ema, epoch, batch_idx, 4, 2, dict(fm_only_epochs=1))
                values.append(row['kmf_gamma'])
                torch.testing.assert_close(model.kmf_generator.progress, ema.kmf_generator.progress)
        for actual, expected in zip(values, [-.5, -.5, -.5, -.3, -.1, .1, .3, .5]):
            self.assertAlmostEqual(actual, expected)

    def test_resume_uses_absolute_epoch_and_disabled_warmup_starts_immediately(self):
        model = policy()
        row = set_kmf_progress(model, None, 2, 0, 4, 2, dict(fm_only_epochs=1))
        self.assertAlmostEqual(row['kmf_progress'], .4)
        row = set_kmf_progress(model, None, 0, 0, 4, 2, dict(enabled=False))
        self.assertEqual(row['kmf_progress'], 0.)
        row = set_kmf_progress(model, None, 3, 1, 4, 2)
        self.assertEqual(row['kmf_progress'], 1.)
        row = set_kmf_progress(model, None, 2, 0, 5, 1)
        self.assertEqual(row['kmf_progress'], .5)
        self.assertEqual(row['kmf_gamma'], 0.)

    def test_fractional_boundary_and_zero_or_one_mixed_batch(self):
        model = policy()
        row = set_kmf_progress(model, None, 400, 0, 501, 1, dict(fm_only_fraction=.8))
        self.assertEqual(row['kmf_progress'], 0.)
        row = set_kmf_progress(model, None, 401, 0, 501, 1, dict(fm_only_fraction=.8))
        self.assertEqual(row['kmf_progress'], 0.)
        row = set_kmf_progress(model, None, 1, 0, 2, 1, dict(fm_only_epochs=1))
        self.assertEqual(row['kmf_gamma'], -.5)
        row = set_kmf_progress(model, None, 1, 0, 2, 1, dict(fm_only_epochs=2))
        self.assertEqual(row['kmf_gamma'], -.5)
        model.flow_ratio = 1.
        row = set_kmf_progress(model, None, 3, 1, 4, 2)
        self.assertEqual(row['kmf_progress'], 0.)

    def test_baseline_and_midpoint_are_noops(self):
        for mode in ('meanflow', 'kmf_midpoint'):
            model = SimpleNamespace(mf_target_type=mode)
            self.assertEqual(set_kmf_progress(model, None, 1, 0, 2, 1), {})

    def test_invalid_training_position_is_rejected(self):
        for epoch, batch, epochs, batches in ((-1, 0, 2, 1), (2, 0, 2, 1),
                                             (0, 1, 2, 1), (0, 0, 2, 0)):
            with self.subTest(epoch=epoch, batch=batch, batches=batches), self.assertRaises(ValueError):
                set_kmf_progress(policy(), None, epoch, batch, epochs, batches)


if __name__ == '__main__':
    unittest.main()
