"""Analytic targets and gradient isolation for learned K-MF interval splits."""

import ast
import math
from pathlib import Path
from types import MethodType
from typing import Tuple
import unittest
from unittest.mock import patch

import torch

from source.common.kinematic_mean_flow import (
    KinematicSplitGenerator,
    add_kmf_optimizer_group,
    learnable_target,
    midpoint_target,
)
from source.common.progressive_gap import isolated_rng
from test_kinematic_mean_flow import NonlinearVelocity
from test_mean_flow_schedule import POLICY_FILES, training_fixture


class ConstantSplit(torch.nn.Module):
    def __init__(self, value):
        super().__init__()
        self.logit = torch.nn.Parameter(torch.tensor(math.log(value / (1 - value)), dtype=torch.float64))

    def forward(self, t, r):
        return self.logit.sigmoid().expand_as(t)


def analytic_derivatives(model, actions, noise, t, r, cond, split):
    """Differentiate the test field algebraically with detached JVP locations."""
    with torch.no_grad():
        upper = t.reshape(t.shape[0], -1, 1)
        lower = r.reshape(r.shape[0], -1, 1)
        center = lower + split.detach().reshape_as(upper) * (upper - lower)
        velocity = noise - actions
        state_t = (1 - upper) * actions + upper * noise
        state_c = (1 - center) * actions + center * noise
        condition = cond.mean((1, 2), keepdim=True)
        d_tc = model.scale * ((2 * state_t + .1 * condition) * velocity + .6 * upper + .4 * center)
        d_cr = model.scale * ((2 * state_c + .1 * condition) * velocity + .6 * center + .4 * lower)
        d_tr = model.scale * ((2 * state_t + .1 * condition) * velocity + .6 * upper + .4 * lower)
    return d_tc, d_cr, d_tr


class LearnableTargetTests(unittest.TestCase):
    def setUp(self):
        self.actions = torch.linspace(-.3, .7, 24, dtype=torch.float64).reshape(3, 4, 2)
        self.noise = torch.linspace(.8, -.5, 24, dtype=torch.float64).reshape(3, 4, 2)
        self.cond = torch.linspace(-.2, .4, 18, dtype=torch.float64).reshape(3, 2, 3)
        self.model = NonlinearVelocity().double()

    def test_target_auxiliary_and_generator_gradient_match_analytic_scalar_and_vector_fields(self):
        times = (
            (torch.tensor([.8, .6, .9], dtype=torch.float64),
             torch.tensor([.1, .6, .3], dtype=torch.float64)),
            (torch.tensor([[.8, .7, .6, .5], [.6, .5, .4, .3], [.4, .6, .8, .9]], dtype=torch.float64),
             torch.tensor([[.1, .2, .3, .4], [.6, .5, .4, .3], [.2, .2, .2, .2]], dtype=torch.float64)),
        )
        for t, r in times:
            for gamma in (-.5, 0., .5):
                with self.subTest(time_shape=t.shape, gamma=gamma):
                    generator = ConstantSplit(.3)
                    tb = t.reshape(3, -1, 1)
                    z = (1 - tb) * self.actions + tb * self.noise
                    prediction = self.model(z, t, r, self.cond)
                    actual_prediction, target, auxiliary, diagnostics = learnable_target(
                        self.model, self.actions, self.noise, t, r, self.cond, generator, gamma)
                    split = torch.full_like(t, .3, requires_grad=True)
                    d_tc, d_cr, d_tr = analytic_derivatives(
                        self.model, self.actions, self.noise, t, r, self.cond, split)
                    lam = split.reshape(3, -1, 1)
                    decoupled = (1 - lam).square() * d_tc + lam.square() * d_cr
                    expected_target = self.noise - self.actions - (t - r).reshape(3, -1, 1) * decoupled
                    delta_norm = (decoupled - d_tr).square().sum((1, 2)).sqrt()
                    expected_weights = 1 + gamma * torch.exp(-1 / (delta_norm + 1e-6))
                    residual_mse = (prediction.detach() - expected_target).square().mean((1, 2))
                    expected_auxiliary = (expected_weights * residual_mse).mean()
                    expected_gradient = torch.autograd.grad(expected_auxiliary, split)[0].sum() * .3 * .7
                    actual_gradient = torch.autograd.grad(auxiliary, generator.logit)[0]
                    torch.testing.assert_close(actual_prediction, prediction)
                    torch.testing.assert_close(target, expected_target)
                    torch.testing.assert_close(auxiliary, expected_auxiliary)
                    torch.testing.assert_close(actual_gradient, expected_gradient)
                    self.assertAlmostEqual(diagnostics['kmf_lambda_mean'], .3)

    def test_auxiliary_and_policy_objectives_update_disjoint_parameters(self):
        actions = self.actions.clone().requires_grad_()
        noise = self.noise.clone().requires_grad_()
        cond = self.cond.clone().requires_grad_()
        generator = ConstantSplit(.3)
        t = torch.tensor([.8, .6, .9], dtype=torch.float64)
        r = torch.tensor([.1, .2, .3], dtype=torch.float64)
        prediction, target, auxiliary, _ = learnable_target(
            self.model, actions, noise, t, r, cond, generator, .5)
        parameters = (self.model.scale, cond, actions, noise, generator.logit)
        auxiliary_gradients = torch.autograd.grad(auxiliary, parameters, allow_unused=True)
        self.assertTrue(all(gradient is None for gradient in auxiliary_gradients[:-1]))
        self.assertGreater(auxiliary_gradients[-1].abs().item(), 0.)
        policy_loss = (prediction - target.detach()).square().mean()
        model_gradient, observation_gradient, generator_gradient = torch.autograd.grad(
            policy_loss, (self.model.scale, cond, generator.logit), allow_unused=True)
        self.assertGreater(model_gradient.abs().item(), 0.)
        self.assertGreater(observation_gradient.abs().sum().item(), 0.)
        self.assertIsNone(generator_gradient)

    def test_zero_gap_target_is_constant_but_auxiliary_weight_can_train_generator(self):
        t = torch.tensor([.8, .6, .9], dtype=torch.float64)
        r = t.clone()
        generator = ConstantSplit(.3)
        _, target, auxiliary, _ = learnable_target(
            self.model, self.actions, self.noise, t, r, self.cond, generator, .5)
        torch.testing.assert_close(target, self.noise - self.actions)
        target_gradient = torch.autograd.grad(target.sum(), generator.logit, retain_graph=True)[0]
        auxiliary_gradient = torch.autograd.grad(auxiliary, generator.logit)[0]
        self.assertEqual(target_gradient.item(), 0.)
        self.assertGreater(auxiliary_gradient.abs().item(), 0.)
        _, _, unweighted_loss, _ = learnable_target(
            self.model, self.actions, self.noise, t, r, self.cond, generator, 0.)
        unweighted_gradient = torch.autograd.grad(unweighted_loss, generator.logit)[0]
        self.assertEqual(unweighted_gradient.item(), 0.)

    def test_half_precision_derivatives_and_residuals_are_promoted_before_arithmetic(self):
        actions = torch.zeros(2, 3, 2, dtype=torch.float16)
        noise = torch.ones_like(actions)
        prediction = torch.full_like(actions, 40000.)
        cond = torch.zeros(2, 2, 3, dtype=torch.float16)
        t = torch.tensor([1., .5])
        r = torch.tensor([0., .5])
        whole_derivative = torch.full((2, 3, 2), 40000., dtype=torch.float16)
        subinterval_derivatives = torch.full((4, 3, 2), 40000., dtype=torch.float16)
        generator = ConstantSplit(.3).float()
        with patch.object(torch.autograd.functional, 'jvp', side_effect=[
                (prediction, whole_derivative), (None, subinterval_derivatives)]):
            _, target, auxiliary, _ = learnable_target(
                self.model, actions, noise, t, r, cond, generator, .5)
        gradient = torch.autograd.grad(auxiliary, generator.logit)[0]
        self.assertEqual(target.dtype, torch.float32)
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.isfinite(auxiliary))
        self.assertTrue(torch.isfinite(gradient))
        torch.testing.assert_close(target[0], torch.full((3, 2), -23199.))
        torch.testing.assert_close(target[1], torch.ones(3, 2))

    def test_initial_generator_matches_midpoint_and_stays_float32_under_cpu_autocast(self):
        generator = KinematicSplitGenerator()
        t = torch.tensor([[.8, .7, .6, .5], [.6, .5, .4, .3], [.4, .6, .8, .9]])
        r = torch.tensor([[.1, .2, .3, .4], [.6, .5, .4, .3], [.2, .2, .2, .2]])
        with torch.autocast('cpu', dtype=torch.bfloat16):
            split = generator(t, r)
        self.assertEqual(split.dtype, torch.float32)
        torch.testing.assert_close(split, torch.full_like(t, .5))
        _, target, auxiliary, _ = learnable_target(
            self.model, self.actions, self.noise, t, r, self.cond, generator, -.5)
        fixed = midpoint_target(self.model, self.actions, self.noise, t, r, self.cond)
        torch.testing.assert_close(target, fixed)
        auxiliary.backward()
        self.assertGreater(generator.network[-2].weight.grad.abs().sum().item(), 0.)

    def test_validation_under_no_grad_returns_finite_graph_free_outputs(self):
        t = torch.tensor([.8, .6, .9], dtype=torch.float64)
        r = torch.tensor([.1, .2, .3], dtype=torch.float64)
        with torch.no_grad():
            prediction, target, auxiliary, _ = learnable_target(
                self.model, self.actions, self.noise, t, r, self.cond,
                ConstantSplit(.3), .5)
        self.assertFalse(prediction.requires_grad)
        self.assertFalse(target.requires_grad)
        self.assertFalse(auxiliary.requires_grad)
        self.assertTrue(torch.isfinite(target).all())
        self.assertTrue(torch.isfinite(auxiliary))

    def test_whole_interval_reference_uses_prediction_dropout_realization(self):
        class DropoutField(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor(.4, dtype=torch.float64))

            def forward(self, z, t, r, cond=None):
                return self.scale * torch.nn.functional.dropout(z.square(), p=.5, training=True)

        model = DropoutField()
        actions = self.actions.abs() + .2
        noise = self.noise.abs() + .4
        t = torch.tensor([.8, .6, .9], dtype=torch.float64)
        r = torch.tensor([.1, .2, .3], dtype=torch.float64)
        with isolated_rng(79):
            prediction, target, auxiliary, _ = learnable_target(
                model, actions, noise, t, r, self.cond, ConstantSplit(.3), .5)
        z = (1 - t[:, None, None]) * actions + t[:, None, None] * noise
        velocity = noise - actions
        # For u = scale * mask * z**2, the same-mask whole derivative is
        # exactly 2 * velocity * u / z. All z are positive in this fixture.
        whole_derivative = 2 * velocity * prediction.detach() / z
        decoupled = (velocity - target) / (t - r)[:, None, None]
        delta_norm = torch.linalg.vector_norm((decoupled - whole_derivative).flatten(1), dim=1)
        weights = 1 + .5 * torch.exp(-1 / (delta_norm + 1e-6))
        residual = (prediction.detach() - target).square().mean((1, 2))
        torch.testing.assert_close(auxiliary, (weights * residual).mean())


class LearnableOptimizerTests(unittest.TestCase):
    def test_production_optimizers_train_generator_and_pure_fm_skips_it_in_all_policies(self):
        with isolated_rng(67):
            batch = dict(action=torch.randn(10, 6, 2), obs=torch.randn(10, 6, 3))
        for filename in POLICY_FILES:
            with self.subTest(policy=filename), isolated_rng(71):
                policy = training_fixture(filename, .2, 'kmf_learnable')
                policy.model = NonlinearVelocity()
                policy.model.get_optim_groups = lambda weight_decay: [
                    {'params': list(policy.model.parameters()), 'weight_decay': weight_decay}]
                path = Path(__file__).resolve().parents[1] / 'source/policy' / filename
                tree = ast.parse(path.read_text(encoding='utf-8'))
                cls = next(node for node in tree.body if isinstance(node, ast.ClassDef))
                optimizer_method = next(node for node in cls.body
                                        if isinstance(node, ast.FunctionDef) and node.name == 'get_optimizer')
                namespace = dict(torch=torch, Tuple=Tuple, add_kmf_optimizer_group=add_kmf_optimizer_group)
                exec(compile(ast.Module(body=[optimizer_method], type_ignores=[]), str(path), 'exec'), namespace)
                policy.get_optimizer = MethodType(namespace['get_optimizer'], policy)
                optimizer_args = dict(transformer_weight_decay=.01, learning_rate=.01, betas=(.9, .99))
                if '_image_' in filename:
                    policy.obs_encoder = torch.nn.Linear(3, 3)
                    policy.process_obs = lambda obs: policy.obs_encoder(obs[:, :2]).reshape(-1, 3)
                    optimizer_args['obs_encoder_weight_decay'] = .01
                optimizer = policy.get_optimizer(**optimizer_args)
                optimizer_ids = [id(parameter) for group in optimizer.param_groups for parameter in group['params']]
                generator_parameters = list(policy.kmf_generator.parameters())
                for parameter in generator_parameters:
                    self.assertEqual(optimizer_ids.count(id(parameter)), 1)
                generator_groups = [group for group in optimizer.param_groups
                                    if any(id(parameter) == id(generator_parameters[0]) for parameter in group['params'])]
                self.assertEqual(generator_groups[0]['weight_decay'], 0.)

                before = [parameter.detach().clone() for parameter in generator_parameters]
                loss = policy.compute_loss(batch)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                self.assertTrue(any(not torch.equal(old, new) for old, new in zip(before, generator_parameters)))

                policy.flow_ratio = 1.
                before_fm = [parameter.detach().clone() for parameter in generator_parameters]
                with patch.object(torch.autograd.functional, 'jvp', side_effect=AssertionError('Pure FM calls no JVP')):
                    fm_loss = policy.compute_loss(batch)
                fm_loss.backward()
                self.assertTrue(all(parameter.grad is None for parameter in generator_parameters))
                optimizer.step()
                self.assertTrue(all(torch.equal(old, new) for old, new in zip(before_fm, generator_parameters)))
                self.assertEqual(policy.kmf_diagnostics, {})


if __name__ == '__main__':
    unittest.main()
