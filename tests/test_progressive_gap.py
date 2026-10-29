"""CPU checks of the actual AMF sampler/loss and batch-based curriculum."""

import copy
import unittest
from unittest.mock import patch

import numpy as np
import torch

from source.common.progressive_gap import (
    GapWeightingSchedule, aggregate_diagnostics, calibrate_gap_lambda,
    capture_rng, epoch_batch_sizes, gap_config, gap_loss, gap_weights,
    isolated_rng, restore_rng,
)
from test_mean_flow_schedule import training_fixture


def amf(ratio=0.2):
    policy = training_fixture('adaptive_mean_flow_transformer_lowdim_policy.py', ratio)
    policy.compute_loss.__func__.__globals__['gap_loss'] = gap_loss
    return policy


class WeightTests(unittest.TestCase):
    def setUp(self):
        self.t = torch.tensor([[0.5, 0.5], [0.1, 0.1], [0.5, 0.5], [0.9, 0.9]])
        self.r = torch.tensor([[0.5, 0.5], [0., 0.], [0., 0.], [0., 0.]])
        self.mask = torch.tensor([True, False, False, False])

    def test_endpoints_mask_direction_and_k(self):
        early = gap_weights(self.t, self.r, self.mask, 0., 1.6)
        torch.testing.assert_close(early, torch.tensor([1., 1.44, .8, .16]))
        torch.testing.assert_close(gap_weights(self.t, self.r, self.mask, 1., 1.6), torch.ones(4))
        middle = gap_weights(self.t, self.r, self.mask, .5, 1.6)
        slower = gap_weights(self.t, self.r, self.mask, .5, 1.6, k=2.)
        torch.testing.assert_close(middle, (early + 1) / 2)
        self.assertGreater(slower[1], middle[1])
        self.assertLess(slower[3], middle[3])

    def test_external_weight_scales_each_sample_gradient_exactly(self):
        error = torch.arange(1., 17.).reshape(4, 2, 2).requires_grad_()
        base = gap_loss(error, self.t, self.r, self.mask)
        base_grad = torch.autograd.grad(base, error)[0]
        context = dict(progress=.3, gap_lambda=1.6, k=1.)
        weighted = gap_loss(error, self.t, self.r, self.mask, context)
        weighted_grad = torch.autograd.grad(weighted, error)[0]
        weight = gap_weights(self.t, self.r, self.mask, **context)
        torch.testing.assert_close(weighted_grad, base_grad * weight[:, None, None])

    def test_small_gap_mf_does_not_become_fm(self):
        t = torch.tensor([[1e-7, 1e-7]])
        w = gap_weights(t, torch.zeros_like(t), torch.tensor([False]), 0., 1.6)
        self.assertAlmostEqual(w.item(), 1.6, places=5)

    def test_uses_mean_trajectory_gap_and_full_batch_mean(self):
        error = torch.tensor([[[1.], [10.]], [[3.], [3.]]])
        t = torch.tensor([[0., 1.], [.2, .2]])
        mask = torch.tensor([False, True])
        r = torch.tensor([[0., 0.], [.2, .2]])
        loss = gap_loss(error, t, r, mask, dict(progress=0., gap_lambda=1.6))
        mse = error.square().mean((1, 2))
        expected = (torch.tensor([.8, 1.]) * mse / (mse + .001).sqrt()).mean()
        torch.testing.assert_close(loss, expected)

    def test_half_precision_reduction_is_finite(self):
        error = torch.full((4, 2, 3), 400., dtype=torch.float16, requires_grad=True)
        with torch.autocast('cpu', dtype=torch.bfloat16):
            loss = gap_loss(error, self.t, self.r, self.mask, dict(progress=.5, gap_lambda=1.6))
        loss.backward()
        self.assertEqual(loss.dtype, torch.float32)
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(error.grad).all())

    def test_invalid_times_shapes_and_parameters(self):
        for context in (dict(progress=float('nan'), gap_lambda=1.), dict(progress=0., gap_lambda=0.),
                        dict(progress=0., gap_lambda=1., k=-1.)):
            with self.assertRaises(ValueError):
                gap_weights(self.t, self.r, self.mask, **context)
        with self.assertRaises(ValueError):
            gap_weights(self.r, self.t, self.mask, 0., 1.6)
        with self.assertRaises(ValueError):
            gap_weights(self.t, self.r, self.mask.float(), 0., 1.6)


class SamplerAndLossTests(unittest.TestCase):
    def test_optional_mask_preserves_samples_rng_and_integer_fm_count(self):
        for size in (1, 2, 10, 256):
            for ratio in (0., .2, 1.):
                policy = amf(ratio)
                with isolated_rng(42):
                    t, r = policy.sample_t_r(size, 'cpu')
                    after = np.random.rand()
                with isolated_rng(42):
                    tm, rm, mask = policy.sample_t_r(size, 'cpu', return_mask=True)
                    self.assertEqual(after, np.random.rand())
                self.assertTrue(torch.equal(t, tm) and torch.equal(r, rm))
                self.assertEqual(int(mask.sum()), int(size * ratio))
                self.assertTrue((t[mask] == r[mask]).all())

    def test_disabled_loss_and_gradient_identical_with_or_without_diagnostics(self):
        policy = amf()
        batch = dict(action=torch.randn(10, 6, 2), obs=torch.randn(10, 6, 3))
        with isolated_rng(12):
            loss = policy.compute_loss(batch)
            grad = torch.autograd.grad(loss, policy.model.scale)[0]
            state = capture_rng()
        with isolated_rng(12):
            other, details = policy.compute_loss(batch, return_diagnostics=True)
            other_grad = torch.autograd.grad(other, policy.model.scale)[0]
            self.assertTrue(torch.equal(state['torch'], capture_rng()['torch']))
        self.assertTrue(torch.equal(loss, other))
        self.assertTrue(torch.equal(grad, other_grad))
        self.assertEqual(details['fm_count'], 2)

    def test_pure_fm_keeps_exact_loss_skips_jvp_and_avoids_empty_means(self):
        policy = amf(1.)
        batch = dict(action=torch.randn(1, 6, 2), obs=torch.randn(1, 6, 3))
        with isolated_rng(1):
            base = policy.compute_loss(batch)
        with isolated_rng(1), patch.object(torch.autograd.functional, 'jvp', side_effect=AssertionError):
            loss, details = policy.compute_loss(batch, dict(progress=0., gap_lambda=1.6), True)
        self.assertTrue(torch.equal(base, loss))
        self.assertEqual(details['mf_count'], 0)
        self.assertNotIn('mf_weight_mean', details)

    def test_pure_mf_batch_one_has_finite_gradients_and_one_jvp(self):
        policy = amf(0.)
        batch = dict(action=torch.randn(1, 6, 2), obs=torch.randn(1, 6, 3))
        with patch.object(torch.autograd.functional, 'jvp', wraps=torch.autograd.functional.jvp) as jvp:
            loss, details = policy.compute_loss(batch, dict(progress=0., gap_lambda=1.6), True)
        jvp.assert_called_once()
        loss.backward()
        self.assertEqual(details['fm_count'], 0)
        self.assertTrue(torch.isfinite(policy.model.scale.grad))


class CalibrationTests(unittest.TestCase):
    def test_cpu_calibration_does_not_queue_a_lazy_cuda_seed(self):
        with patch.object(torch.cuda, 'is_initialized', return_value=False), \
                patch.object(torch.cuda, 'manual_seed_all') as seed_cuda:
            calibrate_gap_lambda(amf(), [10], .2, batches=2)
        seed_cuda.assert_not_called()

    def test_calibration_restores_all_random_streams_and_policy_ratio(self):
        policy = amf(1.)
        with isolated_rng(987):
            state = capture_rng()
            expected = (np.random.rand(), torch.rand(3))
            restore_rng(state)
            value = calibrate_gap_lambda(policy, [256, 17], .2)
            actual = (np.random.rand(), torch.rand(3))
        self.assertEqual(actual[0], expected[0])
        self.assertTrue(torch.equal(actual[1], expected[1]))
        self.assertEqual(policy.flow_ratio, 1.)
        self.assertTrue(1.2 < value < 2.)

    def test_excludes_fm_from_calibration(self):
        policy = amf()
        t = torch.tensor([[1., 1.], [.5, .5], [.5, .5], [.5, .5]])
        r = torch.tensor([[1., 1.], [0., 0.], [0., 0.], [0., 0.]])
        with patch.object(policy, 'sample_t_r', return_value=(t, r, torch.tensor([True, False, False, False]))):
            self.assertEqual(calibrate_gap_lambda(policy, [4], .25, batches=2), 2.)

    def test_mean_weight_near_one_over_independent_mf_samples(self):
        policy = amf(.2)
        value = calibrate_gap_lambda(policy, [256], .2)
        with isolated_rng(123):
            t, r, mask = policy.sample_t_r(20000, 'cpu', return_mask=True)
        weights = gap_weights(t, r, mask, 0., value)
        self.assertAlmostEqual(weights[~mask].mean().item(), 1., delta=.015)


class ProgressTests(unittest.TestCase):
    def schedule(self, epochs=800, start=400, sizes=None, next_epoch=0, saved_state=None, **config):
        return GapWeightingSchedule(dict(enabled=True, lambda_value=1.6, **config),
                                    amf(), epochs, sizes or [10, 3], .2,
                                    dict(fm_only_epochs=start), next_epoch, saved_state)

    def test_400_plus_400_endpoints_and_validation_does_not_consume(self):
        schedule = self.schedule()
        for epoch in range(400):
            for _ in range(2):
                schedule.consume(epoch)
        self.assertEqual(schedule.context()['progress'], 0.)
        for _ in range(10):
            schedule.context()
        self.assertEqual(schedule.state['joint_train_batches_seen'], 0)
        for epoch in range(400, 800):
            for i in range(2):
                q = schedule.context()['progress']
                schedule.consume(epoch)
        self.assertEqual(q, 1.)
        self.assertEqual(schedule.state['joint_train_batches_seen'], 800)

    def test_default_501_boundary_and_absolute_resume(self):
        schedule = GapWeightingSchedule(dict(enabled=True, lambda_value=1.6), amf(),
                                        501, [10], .2, dict(fm_only_fraction=.8), 401)
        self.assertEqual(schedule.state['plan']['joint_start_epoch'], 401)
        self.assertEqual(schedule.context()['progress'], 0.)
        schedule.consume(401)
        restored = GapWeightingSchedule(dict(enabled=True, lambda_value=1.6), amf(),
                                        501, [10], .2, dict(fm_only_fraction=.8), 402, schedule.state)
        self.assertEqual(restored.context(), schedule.context())

    def test_checkpoint_preserves_calibrated_lambda_without_recalibrating(self):
        schedule = GapWeightingSchedule(dict(enabled=True), amf(), 4, [10], .2, dict(fm_only_epochs=1))
        schedule.consume(1)
        with patch('source.common.progressive_gap.calibrate_gap_lambda', side_effect=AssertionError):
            restored = GapWeightingSchedule(dict(enabled=True), amf(), 4, [10], .2,
                                            dict(fm_only_epochs=1), 2, schedule.state)
        self.assertEqual(restored.state, schedule.state)
        self.assertIsNot(restored.state, schedule.state)

    def test_invalid_resume_replay_and_changed_plan_rejected(self):
        schedule = self.schedule()
        schedule.consume(400)
        schedule.consume(400)
        for kwargs in (dict(next_epoch=400), dict(next_epoch=401, epochs=801),
                       dict(next_epoch=401, k=2.)):
            with self.assertRaises(ValueError):
                self.schedule(saved_state=schedule.state, **kwargs)

    def test_zero_one_and_constant_mode(self):
        with patch('source.common.progressive_gap.calibrate_gap_lambda', side_effect=AssertionError):
            empty = self.schedule(epochs=2, start=2)
            self.assertIsNone(empty.context())
            only_fm = GapWeightingSchedule(dict(enabled=True), amf(), 2, [10], 1.)
            self.assertIsNone(only_fm.context())
        single = self.schedule(epochs=2, start=1, sizes=[1])
        self.assertEqual(single.context()['progress'], 1.)
        constant = self.schedule(epochs=2, start=1, sizes=[1], mode='constant_small_gap')
        self.assertEqual(constant.context()['progress'], 0.)

    def test_capped_and_partial_batches(self):
        self.assertEqual(epoch_batch_sizes(13, 5), [5, 5, 3])
        self.assertEqual(epoch_batch_sizes(13, 5, True), [5, 5])
        self.assertEqual(epoch_batch_sizes(13, 5, max_train_steps=2), [5, 5])
        with self.assertRaises(ValueError):
            epoch_batch_sizes(1, 5, True)

    def test_config_validation_and_default_off(self):
        self.assertFalse(gap_config()['enabled'])
        for bad in (dict(k=0), dict(k=float('nan')), dict(lambda_value=float('inf')),
                    dict(aggregation='position'), dict(start_at='training_start'),
                    dict(mode='unknown'), dict(calibration_batches=0), dict(enabled=1)):
            with self.assertRaises(ValueError):
                gap_config(bad)

    def test_validation_means_use_counts_and_omit_empty_groups(self):
        rows = [dict(fm_count=1, fm_mse=3., mf_count=0), dict(fm_count=3, fm_mse=1., mf_count=0)]
        result = aggregate_diagnostics(rows)
        self.assertEqual(result['fm_mse'], 1.5)
        self.assertNotIn('mf_mse', result)


if __name__ == '__main__':
    unittest.main()
