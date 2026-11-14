"""Fixed-midpoint targets, detached gradients and the four policy entry points."""

import unittest
from unittest.mock import patch

import torch

from source.common.kinematic_mean_flow import midpoint_target
from source.common.progressive_gap import gap_loss, gap_weights, isolated_rng
from test_mean_flow_schedule import POLICY_FILES, training_fixture


class NonlinearVelocity(torch.nn.Module):
    """An analytic field that distinguishes both times and intermediate states."""

    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(.4))

    def forward(self, z, t, r, cond=None):
        t = t.reshape(t.shape[0], -1, 1)
        r = r.reshape(r.shape[0], -1, 1)
        condition = cond.mean((1, 2), keepdim=True)
        return self.scale * (z.square() + .3 * t.square() + .2 * r
                             + .4 * t * r + .1 * condition * z)


def analytic_target(model, actions, noise, t, r, cond):
    # Differentiate NonlinearVelocity algebraically. This deliberately uses no
    # production helper or autograd derivative to check the JVP construction.
    with torch.no_grad():
        t = t.reshape(t.shape[0], -1, 1)
        r = r.reshape(r.shape[0], -1, 1)
        c = (t + r) / 2
        zt = (1 - t) * actions + t * noise
        zc = (1 - c) * actions + c * noise
        velocity = noise - actions
        condition = cond.mean((1, 2), keepdim=True)
        early = model.scale * ((2 * zt + .1 * condition) * velocity + .6 * t + .4 * c)
        late = model.scale * ((2 * zc + .1 * condition) * velocity + .6 * c + .4 * r)
        return velocity - (t - r) * (early + late) / 4


def adaptive_loss(error):
    mse = error.square().mean((1, 2))
    weight = (mse.detach() + .001).rsqrt()
    return (mse * weight).mean()


class MidpointTargetTests(unittest.TestCase):
    def test_half_precision_derivatives_are_promoted_before_summing(self):
        derivatives = torch.full((4, 3, 2), 40000., dtype=torch.float16)
        t, r = torch.tensor([1., .5]), torch.tensor([0., .5])
        for dtype in (torch.float16, torch.float32):
            with self.subTest(action_dtype=dtype):
                actions = torch.zeros(2, 3, 2, dtype=dtype)
                noise = torch.ones_like(actions)
                cond = torch.zeros(2, 2, 3, dtype=dtype)
                with patch.object(torch.autograd.functional, 'jvp', return_value=(None, derivatives)):
                    target = midpoint_target(NonlinearVelocity(), actions, noise, t, r, cond)
                self.assertEqual(target.dtype, torch.float32)
                self.assertTrue(torch.isfinite(target).all())
                torch.testing.assert_close(target[0], torch.full((3, 2), -19999.))
                torch.testing.assert_close(target[1], torch.ones(3, 2))

    def test_scalar_and_vector_targets_match_analytic_derivatives_without_graph(self):
        with isolated_rng(7):
            actions = torch.randn(3, 4, 2, dtype=torch.float64, requires_grad=True)
            noise = torch.randn_like(actions)
            cond = torch.randn(3, 2, 3, dtype=torch.float64, requires_grad=True)
        model = NonlinearVelocity().double()
        times = (
            (torch.tensor([.8, .6, .4], dtype=torch.float64),
             torch.tensor([.1, .6, .2], dtype=torch.float64)),
            (torch.tensor([[.8, .7, .6, .5], [.6, .5, .4, .3], [.4, .6, .8, 1.]], dtype=torch.float64),
             torch.tensor([[.1, .2, .3, .4], [.6, .5, .4, .3], [.2, .2, .2, .2]], dtype=torch.float64)),
        )
        for t, r in times:
            with self.subTest(time_shape=t.shape), torch.no_grad(), patch.object(
                    torch.autograd.functional, 'jvp', wraps=torch.autograd.functional.jvp) as jvp:
                target = midpoint_target(model, actions, noise, t, r, cond)
                expected = analytic_target(model, actions, noise, t, r, cond)
                torch.testing.assert_close(target, expected)
                torch.testing.assert_close(target[1], (noise - actions)[1])
                self.assertFalse(target.requires_grad)
                jvp.assert_called_once()
                self.assertFalse(jvp.call_args.kwargs.get('create_graph', False))
                self.assertEqual(jvp.call_args.args[1][0].shape[0], 2 * actions.shape[0])


class MidpointPolicyTests(unittest.TestCase):
    def setUp(self):
        with isolated_rng(19):
            self.batch = dict(action=torch.randn(10, 6, 2), obs=torch.randn(10, 6, 3))

    def test_all_policies_match_analytic_target_loss_and_detached_target_gradient(self):
        for filename in POLICY_FILES:
            with self.subTest(policy=filename), isolated_rng(23):
                policy = training_fixture(filename, .2, 'kmf_midpoint')
                policy.model = NonlinearVelocity()
                t, r = policy.sample_t_r(10, 'cpu')
                policy.sample_t_r = lambda batch_size, device: (t, r)
                noise = torch.randn_like(self.batch['action'])
                cond = self.batch['obs'][:, :2]
                target = analytic_target(policy.model, self.batch['action'], noise, t, r, cond)
                tb = t.reshape(10, -1, 1)
                z = (1 - tb) * self.batch['action'] + tb * noise
                expected = adaptive_loss(policy.model(z, t, r, cond) - target)
                expected_grad = torch.autograd.grad(expected, policy.model.scale)[0]
                with patch.object(torch, 'randn_like', return_value=noise):
                    actual = policy.compute_loss(self.batch)
                grad = torch.autograd.grad(actual, policy.model.scale)[0]
                torch.testing.assert_close(actual, expected)
                torch.testing.assert_close(grad, expected_grad)

    def test_pure_fm_is_identical_and_skips_jvp_for_every_policy(self):
        for filename in POLICY_FILES:
            with self.subTest(policy=filename):
                policy = training_fixture(filename, 1.)
                with isolated_rng(5):
                    baseline = policy.compute_loss(self.batch)
                    baseline_grad = torch.autograd.grad(baseline, policy.model.scale)[0]
                policy.mf_target_type = 'kmf_midpoint'
                with isolated_rng(5), patch.object(torch.autograd.functional, 'jvp', side_effect=AssertionError):
                    actual = policy.compute_loss(self.batch)
                    grad = torch.autograd.grad(actual, policy.model.scale)[0]
                self.assertTrue(torch.equal(actual, baseline))
                self.assertTrue(torch.equal(grad, baseline_grad))

    def test_single_mf_sample_trains_and_no_grad_validation_works(self):
        batch = {key: value[:1] for key, value in self.batch.items()}
        for filename in POLICY_FILES:
            with self.subTest(policy=filename), isolated_rng(31):
                policy = training_fixture(filename, 0., 'kmf_midpoint')
                loss = policy.compute_loss(batch)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                self.assertTrue(torch.isfinite(policy.model.scale.grad))
                with torch.no_grad():
                    validation_loss = policy.compute_loss(batch)
                self.assertTrue(torch.isfinite(validation_loss))
                self.assertFalse(validation_loss.requires_grad)

    def test_image_observation_encoder_receives_only_prediction_branch_gradients(self):
        for filename in (name for name in POLICY_FILES if '_image_' in name):
            with self.subTest(policy=filename), isolated_rng(37):
                policy = training_fixture(filename, .2, 'kmf_midpoint')
                policy.model = NonlinearVelocity()
                encoder = torch.nn.Linear(3, 3, bias=False)
                policy.process_obs = lambda obs: encoder(obs[:, :2]).reshape(-1, 3)
                t, r = policy.sample_t_r(10, 'cpu')
                policy.sample_t_r = lambda batch_size, device: (t, r)
                noise = torch.randn_like(self.batch['action'])
                cond = encoder(self.batch['obs'][:, :2])
                target = analytic_target(policy.model, self.batch['action'], noise, t, r, cond)
                tb = t.reshape(10, -1, 1)
                z = (1 - tb) * self.batch['action'] + tb * noise
                expected = adaptive_loss(policy.model(z, t, r, cond) - target)
                expected_grad = torch.autograd.grad(expected, encoder.weight)[0]
                with patch.object(torch, 'randn_like', return_value=noise):
                    actual = policy.compute_loss(self.batch)
                grad = torch.autograd.grad(actual, encoder.weight)[0]
                self.assertGreater(grad.abs().sum().item(), 0.)
                torch.testing.assert_close(grad, expected_grad)

    def test_gap_weights_apply_to_midpoint_residual_with_original_sample_mask(self):
        policy = training_fixture('adaptive_mean_flow_transformer_lowdim_policy.py', .2, 'kmf_midpoint')
        policy.compute_loss.__func__.__globals__['gap_loss'] = gap_loss
        policy.model = NonlinearVelocity()
        with isolated_rng(41):
            t, r, mask = policy.sample_t_r(10, 'cpu', return_mask=True)
            noise = torch.randn_like(self.batch['action'])
        policy.sample_t_r = lambda batch_size, device, return_mask=False: (t, r, mask)
        context = dict(progress=.3, gap_lambda=1.6, k=1.)
        cond = self.batch['obs'][:, :2]
        target = analytic_target(policy.model, self.batch['action'], noise, t, r, cond)
        z = (1 - t[..., None]) * self.batch['action'] + t[..., None] * noise
        error = policy.model(z, t, r, cond) - target
        mse = error.square().mean((1, 2))
        weights = gap_weights(t, r, mask, **context)
        expected = (weights * mse * (mse.detach() + .001).rsqrt()).mean()
        expected_grad = torch.autograd.grad(expected, policy.model.scale)[0]
        with patch.object(torch, 'randn_like', return_value=noise):
            actual, details = policy.compute_loss(self.batch, context, True)
        grad = torch.autograd.grad(actual, policy.model.scale)[0]
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(grad, expected_grad)
        self.assertEqual(details['fm_count'], 2)
        self.assertEqual(details['mf_count'], 8)


if __name__ == '__main__':
    unittest.main()
