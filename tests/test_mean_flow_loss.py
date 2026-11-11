"""MF-only Cauchy gradients, unchanged FM/baseline, and gap composition."""

import unittest
from unittest.mock import patch

import torch

from source.common.mean_flow_loss import cauchy_mf_loss, validate_mf_loss
from source.common.progressive_gap import (
    aggregate_diagnostics, gap_loss, gap_weights, isolated_rng,
)
from test_mean_flow_schedule import POLICY_FILES, training_fixture


class CauchyGradientTests(unittest.TestCase):
    def test_per_trajectory_objective_and_analytic_gradient(self):
        error = torch.tensor([[[.1, .2], [.3, .4]],
                              [[.2, 2.], [3., 4.]],
                              [[5., 6.], [7., 8.]]], dtype=torch.float64, requires_grad=True)
        mask = torch.tensor([True, False, False])
        c = .07
        mse = error.detach().square().mean((1, 2))
        expected_loss = (mse[0] / (mse[0] + .001).sqrt()
                         + torch.log1p(mse[1:] / c).sum()) / 3
        loss = cauchy_mf_loss(error, mask, c)
        torch.testing.assert_close(loss, expected_loss)
        grad = torch.autograd.grad(loss, error)[0]
        coefficients = torch.tensor([1 / (mse[0] + .001).sqrt(),
                                     1 / (mse[1] + c), 1 / (mse[2] + c)])
        torch.testing.assert_close(grad, 2 * error.detach() * coefficients[:, None, None] / 12)

    def test_changing_c_only_changes_mf_values_and_gradients(self):
        error = torch.arange(1., 13.).reshape(3, 2, 2).requires_grad_()
        mask = torch.tensor([True, False, True])
        small = torch.autograd.grad(cauchy_mf_loss(error, mask, .001), error)[0]
        large = torch.autograd.grad(cauchy_mf_loss(error, mask, 10.), error)[0]
        self.assertTrue(torch.equal(small[mask], large[mask]))
        self.assertFalse(torch.equal(small[~mask], large[~mask]))

    def test_zero_large_half_precision_errors_and_empty_groups(self):
        for mask in (torch.tensor([True, False]), torch.tensor([False, False]),
                     torch.tensor([True, True])):
            error = torch.zeros(2, 2, 3, dtype=torch.float16)
            error[~mask] = 400.
            error.requires_grad_()
            loss = cauchy_mf_loss(error, mask, .01)
            loss.backward()
            self.assertEqual(loss.dtype, torch.float32)
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(error.grad).all())
            self.assertTrue(torch.equal(error.grad[mask], torch.zeros_like(error.grad[mask])))

    def test_invalid_mode_scale_and_mask_are_rejected(self):
        for mode, c in [('unknown', .01), ('cauchy', 0), ('cauchy', -1),
                        ('cauchy', float('nan')), ('cauchy', float('inf')),
                        ('adaptive_l2', True), ('cauchy', '0.01')]:
            with self.subTest(mode=mode, c=c), self.assertRaises(ValueError):
                validate_mf_loss(mode, c)
        for mask in (torch.ones(2), torch.ones(2, 1, dtype=torch.bool)):
            with self.assertRaises(ValueError):
                cauchy_mf_loss(torch.ones(2, 3, 2), mask, .01)


class GapCompositionTests(unittest.TestCase):
    def test_gap_multiplies_log_loss_and_gradient_outside_log(self):
        error = torch.arange(1., 17.).reshape(4, 2, 2).requires_grad_()
        t = torch.tensor([[.5, .5], [.1, .3], [.5, .5], [.9, .9]])
        r = torch.tensor([[.5, .5], [0., 0.], [0., 0.], [0., 0.]])
        mask = torch.tensor([True, False, False, False])
        context = dict(progress=.3, gap_lambda=1.6, k=1.)
        c = .1
        base = gap_loss(error, t, r, mask, mf_loss_type='cauchy', mf_loss_c=c)
        base_grad = torch.autograd.grad(base, error)[0]
        weighted, details = gap_loss(error, t, r, mask, context, True,
                                     mf_loss_type='cauchy', mf_loss_c=c)
        grad = torch.autograd.grad(weighted, error)[0]
        weights = gap_weights(t, r, mask, **context)
        torch.testing.assert_close(grad, base_grad * weights[:, None, None])
        mse = error.detach().square().mean((1, 2))
        expected = torch.cat([(mse[:1] / (mse[:1] + .001).sqrt()),
                              torch.log1p(mse[1:] / c)])
        torch.testing.assert_close(weighted, (weights * expected).mean())
        baseline = gap_loss(error, t, r, mask, context)
        baseline_grad = torch.autograd.grad(baseline, error)[0]
        self.assertTrue(torch.equal(grad[mask], baseline_grad[mask]))
        self.assertAlmostEqual(details['mf_objective_loss'], expected[1:].mean().item())
        self.assertAlmostEqual(details['mf_weighted_loss'], (weights[1:] * expected[1:]).mean().item())
        self.assertAlmostEqual(details['mf_mse'], mse[1:].mean().item())

    def test_objective_diagnostics_aggregate_by_sample_count(self):
        rows = [dict(fm_count=0, mf_count=1, mf_objective_loss=6.),
                dict(fm_count=0, mf_count=3, mf_objective_loss=2.)]
        result = aggregate_diagnostics(rows)
        self.assertEqual(result['mf_objective_loss'], 3.)
        self.assertNotIn('fm_objective_loss', result)


class PolicyLossTests(unittest.TestCase):
    def setUp(self):
        with isolated_rng(77):
            self.batch = dict(action=torch.randn(10, 6, 2), obs=torch.randn(10, 6, 3))

    def test_default_baseline_and_pure_fm_remain_identical_in_all_policies(self):
        for filename in POLICY_FILES:
            for ratio in (0., .2, 1.):
                policy = training_fixture(filename, ratio)
                with self.subTest(policy=filename, ratio=ratio):
                    with isolated_rng(9):
                        original = policy.compute_loss(self.batch)
                        original_grad = torch.autograd.grad(original, policy.model.scale)[0]
                    policy.mf_loss_type, policy.mf_loss_c = 'adaptive_l2', .2
                    with isolated_rng(9):
                        baseline = policy.compute_loss(self.batch)
                        grad = torch.autograd.grad(baseline, policy.model.scale)[0]
                    self.assertTrue(torch.equal(original, baseline))
                    self.assertTrue(torch.equal(original_grad, grad))
                    if ratio == 1.:
                        policy.mf_loss_type = 'cauchy'
                        with isolated_rng(9), patch.object(torch.autograd.functional, 'jvp', side_effect=AssertionError):
                            fm = policy.compute_loss(self.batch)
                            fm_grad = torch.autograd.grad(fm, policy.model.scale)[0]
                        self.assertTrue(torch.equal(original, fm))
                        self.assertTrue(torch.equal(original_grad, fm_grad))

    def test_optional_sampler_mask_preserves_rng_and_samples(self):
        for filename in POLICY_FILES:
            for ratio in (0., .2, 1.):
                policy = training_fixture(filename, ratio)
                with self.subTest(policy=filename, ratio=ratio):
                    with isolated_rng(7):
                        t, r = policy.sample_t_r(10, 'cpu')
                        next_times = policy.sample_t_r(10, 'cpu')
                    with isolated_rng(7):
                        tm, rm, mask = policy.sample_t_r(10, 'cpu', return_mask=True)
                        next_mask_times = policy.sample_t_r(10, 'cpu')
                    self.assertTrue(torch.equal(t, tm) and torch.equal(r, rm))
                    self.assertTrue(all(torch.equal(a, b) for a, b in zip(next_times, next_mask_times)))
                    self.assertEqual(mask.sum().item(), int(10 * ratio))

    def test_mf_identity_comes_from_mask_even_with_equal_times(self):
        # An MF row can round to t=r. It must not be silently classified as FM.
        for filename in POLICY_FILES:
            policy = training_fixture(filename, .2)
            policy.mf_loss_type, policy.mf_loss_c = 'cauchy', .07
            t, _, mask = policy.sample_t_r(10, 'cpu', return_mask=True)
            policy.sample_t_r = lambda batch_size, device, return_mask=False: (t, t, mask)
            noise = torch.zeros_like(self.batch['action'])
            z = (1 - t.reshape(10, -1, 1)) * self.batch['action']
            v = -self.batch['action']
            error = policy.model(z, t, t) - v
            expected = cauchy_mf_loss(error, mask, .07)
            expected_grad = torch.autograd.grad(expected, policy.model.scale)[0]
            with self.subTest(policy=filename), patch.object(torch, 'randn_like', return_value=noise):
                loss = policy.compute_loss(self.batch)
                grad = torch.autograd.grad(loss, policy.model.scale)[0]
                torch.testing.assert_close(loss, expected)
                torch.testing.assert_close(grad, expected_grad)

    def test_cauchy_mixed_and_single_mf_batches_use_jvp_and_finite_gradients(self):
        for filename in POLICY_FILES:
            for ratio, size in ((.2, 10), (0., 1)):
                policy = training_fixture(filename, ratio)
                policy.mf_loss_type, policy.mf_loss_c = 'cauchy', .03
                batch = {key: value[:size] for key, value in self.batch.items()}
                with self.subTest(policy=filename, ratio=ratio), patch.object(
                        torch.autograd.functional, 'jvp', wraps=torch.autograd.functional.jvp) as jvp:
                    loss = policy.compute_loss(batch)
                    jvp.assert_called_once()
                    loss.backward()
                    self.assertTrue(torch.isfinite(loss))
                    self.assertTrue(torch.isfinite(policy.model.scale.grad))


if __name__ == '__main__':
    unittest.main()
