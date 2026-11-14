"""Short CPU training smoke using the real workspace loop and AMF Transformer.

Optional Hydra/dataset/simulator integrations are replaced by in-memory adapters.
The sampler, JVP loss, Transformer, AdamW, EMA, DataLoader and checkpoint I/O are
production code. This does not claim a CUDA or MetaWorld closed-loop evaluation.
"""

import ast
import copy
import json
import os
import pathlib
import random
import tempfile
import threading
from types import SimpleNamespace
from typing import Optional
import unittest
from unittest.mock import patch

import dill
import numpy as np
import torch
import tqdm

from source.common.checkpoint_util import TopKCheckpointManager
from source.common.json_logger import JsonLogger
from source.common.mean_flow_schedule import scheduled_flow_ratio
from source.common.progressive_gap import (
    GapWeightingSchedule, aggregate_diagnostics, capture_rng, epoch_batch_sizes,
    gap_config, gap_loss, isolated_rng, restore_rng,
)
from source.common.pytorch_util import dict_apply, optimizer_to
from source.model.diffusion.ema_model import EMAModel
from source.model.mean_flow.transformer_for_adaptive_mean_flow import TransformerForAdaptiveMeanFlow
from test_progressive_gap import amf


ROOT = pathlib.Path(__file__).resolve().parents[1]


class Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError:
            raise AttributeError(name) from None

    __setattr__ = dict.__setitem__


def config(value):
    return Config({k: config(v) for k, v in value.items()}) if isinstance(value, dict) else value


class OmegaAdapter:
    @staticmethod
    def to_container(cfg, **kwargs):
        return cfg

    create = staticmethod(lambda value: value)

    @staticmethod
    def save(value, path):
        pathlib.Path(path).write_text(json.dumps(value), encoding='utf-8')


class Normalizer(dict):
    def normalize(self, batch):
        return batch


class Policy(torch.nn.Module):
    def __init__(self, flow_ratio, mf_target_type='meanflow'):
        super().__init__()
        self.horizon, self.n_obs_steps = 6, 2
        self.flow_ratio = flow_ratio
        self.mf_target_type = mf_target_type
        self.monotone_schedule_ratio = .5
        self.time_dist = ['lognorm', -.4, 1.]
        self.normalizer = Normalizer()
        self.model = TransformerForAdaptiveMeanFlow(
            input_dim=2, output_dim=2, horizon=6, n_obs_steps=2,
            cond_dim=3, n_layer=1, n_head=2, n_emb=8,
            p_drop_emb=0., p_drop_attn=0., time_cond='AdaLNZero')

    sample_t_r = amf().sample_t_r.__func__
    compute_loss = amf().compute_loss.__func__

    def model_call(self, z, t, r, cond):
        return self.model(z, t, r, cond)

    def set_normalizer(self, normalizer):
        self.normalizer = normalizer

    def get_optimizer(self, **kwargs):
        return torch.optim.AdamW(self.parameters(), lr=.001)

    def predict_action(self, obs_dict):
        return {'action_pred': torch.zeros(obs_dict['obs'].shape[0], 6, 2, device=obs_dict['obs'].device)}


class Dataset(torch.utils.data.Dataset):
    def __init__(self):
        self.replay_buffer = dict(action=torch.arange(156.).reshape(13, 6, 2) / 100.)

    def __len__(self):
        return 13

    def __getitem__(self, i):
        return dict(action=self.replay_buffer['action'][i], obs=torch.full((6, 3), i / 13.))

    def get_normalizer(self):
        return Normalizer()

    def get_validation_dataset(self):
        return self


class Runner:
    def run(self, policy):
        return {'test/mean_score': 0.}


def instantiate(cfg, **kwargs):
    if cfg.kind == 'policy':
        return Policy(cfg.flow_ratio, cfg.get('mf_target_type', 'meanflow'))
    if cfg.kind == 'dataset':
        return Dataset()
    if cfg.kind == 'runner':
        return Runner()
    if cfg.kind == 'ema':
        return EMAModel(kwargs['model'])
    raise AssertionError(cfg)


def scheduler(name, optimizer, last_epoch, **kwargs):
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: .98 ** step, last_epoch=last_epoch)


def workspace_class():
    # Like the existing tests, load real definitions without importing optional
    # packages. No training-loop, progress, checkpoint or loss logic is copied.
    namespace = dict(globals(), __name__=__name__, OmegaConf=OmegaAdapter,
                     hydra=SimpleNamespace(utils=SimpleNamespace(instantiate=instantiate)),
                     BaseLowdimDataset=Dataset, BaseLowdimRunner=Runner,
                     MeanFlowTransformerLowdimPolicy=Policy,
                     AdaptiveMeanFlowTransformerLowdimPolicy=Policy,
                     SingleFieldLinearNormalizer=SimpleNamespace(create_fit=lambda **kwargs: None),
                     DataLoader=torch.utils.data.DataLoader, get_scheduler=scheduler,
                     wandb=SimpleNamespace(init=lambda **kwargs: SimpleNamespace(log=lambda *a, **kw: None),
                                           config=SimpleNamespace(update=lambda *a, **kw: None)))
    for filename in ('base_workspace.py', 'train_mean_flow_lowdim_workspace.py'):
        path = ROOT / 'source/workspace' / filename
        tree = ast.parse(path.read_text(encoding='utf-8'))
        nodes = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
                 and getattr(n, 'name', '') != 'main']
        exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), namespace)
    return namespace['TrainMeanFlowLowdimWorkspace']


class StopAfterCheckpoint(Exception):
    pass


class WorkspaceSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.Workspace = workspace_class()

    def setUp(self):
        parent = ROOT / 'data/test_progressive_gap'
        parent.mkdir(parents=True, exist_ok=True)
        self.temp = tempfile.TemporaryDirectory(dir=parent)
        self.addCleanup(self.temp.cleanup)
        self.root = pathlib.Path(self.temp.name)
        threads = torch.get_num_threads()
        torch.set_num_threads(1)
        self.addCleanup(torch.set_num_threads, threads)
        self.cfg = config(dict(
            policy=dict(kind='policy', flow_ratio=.2), optimizer={},
            training=dict(device='cpu', seed=42, resume=True, debug=False, use_ema=True,
                          num_epochs=4, max_train_steps=2, max_val_steps=1,
                          gradient_accumulate_every=1, lr_scheduler='test', lr_warmup_steps=0,
                          rollout_every=1, checkpoint_every=1, val_every=1, sample_every=1,
                          tqdm_interval_sec=100, flow_ratio_schedule=dict(fm_only_epochs=1),
                          progressive_gap_weighting=dict(enabled=True, calibration_batches=2)),
            dataloader=dict(batch_size=5, num_workers=0, shuffle=True),
            val_dataloader=dict(batch_size=5, num_workers=0, shuffle=False),
            task=dict(dataset=dict(kind='dataset'), env_runner=dict(kind='runner')),
            ema=dict(kind='ema'), logging={}, inference_sweep=dict(perform_inference_sweep=False),
            checkpoint=dict(topk=dict(monitor_key='epoch', mode='max', k=0),
                            save_last_ckpt=True, save_last_snapshot=False)))

    def make_workspace(self, directory, stop_epoch=None):
        directory.mkdir(exist_ok=True)
        workspace = self.Workspace(copy.deepcopy(self.cfg), output_dir=str(directory))
        original = workspace.save_checkpoint

        def save(*args, **kwargs):
            kwargs['use_thread'] = False
            path = original(*args, **kwargs)
            if workspace.epoch == stop_epoch:
                raise StopAfterCheckpoint()
            return path

        workspace.save_checkpoint = save
        return workspace

    def test_train_validate_save_resume_matches_continuous_next_batch_and_weights(self):
        self.check_train_and_resume()

    def test_midpoint_train_validate_save_resume_preserves_target_and_weights(self):
        self.cfg.policy.mf_target_type = 'kmf_midpoint'
        self.check_train_and_resume()

    def check_train_and_resume(self):
        full = self.make_workspace(self.root / 'continuous')
        full.run()
        interrupted = self.make_workspace(self.root / 'resumed', stop_epoch=1)
        with self.assertRaises(StopAfterCheckpoint):
            interrupted.run()
        self.assertEqual(interrupted.progressive_gap_state['joint_train_batches_seen'], 2)
        resumed = self.make_workspace(self.root / 'resumed')
        with patch('source.common.progressive_gap.calibrate_gap_lambda', side_effect=AssertionError('recalibrated')):
            resumed.run()
        self.assertEqual(resumed.epoch, 4)
        self.assertEqual(resumed.global_step, 8)  # max_train_steps=2, not full dataloader length.
        self.assertEqual(resumed.model.mf_target_type, self.cfg.policy.get('mf_target_type', 'meanflow'))
        self.assertEqual(resumed.ema_model.mf_target_type, resumed.model.mf_target_type)
        with resumed.get_checkpoint_path().open('rb') as stream:
            payload = torch.load(stream, pickle_module=dill, weights_only=False)
        self.assertEqual(payload['cfg'].policy, self.cfg.policy)
        self.assertEqual(resumed.progressive_gap_state, full.progressive_gap_state)
        self.assertEqual(resumed.progressive_gap_state['planned_joint_batches'], 6)
        for key, expected in full.model.state_dict().items():
            torch.testing.assert_close(resumed.model.state_dict()[key], expected, rtol=0, atol=0)
        for key, expected in full.ema_model.state_dict().items():
            torch.testing.assert_close(resumed.ema_model.state_dict()[key], expected, rtol=0, atol=0)
        rows = [json.loads(line) for line in (self.root / 'continuous/logs.json.txt').read_text().splitlines()]
        joint = [r for r in rows if r['epoch'] >= 1]
        self.assertEqual([r['gap_progress'] for r in joint], [0., .2, .4, .6, .8, 1.])
        final = rows[-1]
        self.assertEqual(final['mf_weight_mean'], 1.)
        self.assertIn('val_fm_mse', final)
        self.assertIn('val_mf_mse', final)
        self.assertEqual(sum(final.get(f'val_mf_gap_bin_{i}_count', 0) for i in range(4)), 5)

    def test_debug_limits_used_before_planning_and_zero_joint_is_safe(self):
        self.cfg.training.debug = True
        self.cfg.training.flow_ratio_schedule = config(dict(fm_only_epochs=400))
        workspace = self.make_workspace(self.root / 'debug')
        workspace.run()
        self.assertEqual(workspace.epoch, 2)
        self.assertEqual(workspace.global_step, 6)
        self.assertEqual(workspace.progressive_gap_state['planned_joint_batches'], 0)
        self.assertIsNone(workspace.progressive_gap_state['lambda_value'])

    @unittest.skipUnless(torch.cuda.is_available(), 'CUDA required for real AMP smoke')
    def test_cuda_amp_short_training_has_finite_loss_and_parameters(self):
        try:
            torch.rand(1, device='cuda').item()
        except RuntimeError as exc:
            if 'no kernel image is available' in str(exc):
                self.skipTest('Installed PyTorch has no kernels for this GPU architecture')
            raise
        self.cfg.training.device = 'cuda:0'
        workspace = self.make_workspace(self.root / 'cuda')
        workspace.run()
        self.assertTrue(workspace.enable_amp)
        self.assertEqual(workspace.progressive_gap_state['joint_train_batches_seen'], 6)
        for value in workspace.model.parameters():
            self.assertTrue(torch.isfinite(value).all())
        rows = [json.loads(line) for line in (self.root / 'cuda/logs.json.txt').read_text().splitlines()]
        self.assertTrue(all(np.isfinite(row['train_loss']) for row in rows))
        self.assertEqual(rows[-1]['mf_weight_mean'], 1.)

    def test_unsupported_policy_and_image_workspace_reject_enabled_setting(self):
        with patch.dict(self.Workspace.__init__.__globals__,
                        AdaptiveMeanFlowTransformerLowdimPolicy=type('OtherPolicy', (), {})):
            with self.assertRaisesRegex(ValueError, 'AMF lowdim only'):
                self.make_workspace(self.root / 'unsupported')
        path = ROOT / 'source/workspace/train_mean_flow_transformer_image_workspace.py'
        tree = ast.parse(path.read_text(encoding='utf-8'))
        image_class = next(n for n in tree.body if isinstance(n, ast.ClassDef))
        namespace = dict(self.Workspace.__init__.__globals__)
        exec(compile(ast.Module(body=[image_class], type_ignores=[]), str(path), 'exec'), namespace)
        with self.assertRaisesRegex(ValueError, 'AMF lowdim only'):
            namespace[image_class.name](self.cfg, output_dir=str(self.root))

    def test_old_checkpoint_disabled_loads_and_enabled_recovers_verified_progress(self):
        self.cfg.training.max_train_steps = None
        self.cfg.training.progressive_gap_weighting.enabled = False
        old = self.make_workspace(self.root / 'legacy', stop_epoch=1)
        with self.assertRaises(StopAfterCheckpoint):
            old.run()
        path = old.get_checkpoint_path()
        with path.open('rb') as stream:
            payload = torch.load(stream, pickle_module=dill, weights_only=False)
        payload['pickles'].pop('training_resume_state')
        payload['pickles'].pop('progressive_gap_state')
        # Default-off loading remains compatible with legacy keys.
        disabled = self.make_workspace(self.root / 'legacy-off')
        disabled.load_payload(payload, include_keys=['epoch', 'global_step'])
        self.assertEqual(disabled.epoch, 1)
        self.cfg.training.progressive_gap_weighting.enabled = True
        enabled = self.make_workspace(self.root / 'legacy-enabled')
        enabled.load_payload(payload, include_keys=['epoch', 'global_step'])
        schedule = enabled.prepare_gap_schedule(enabled.cfg, [5, 5, 3])
        self.assertEqual(enabled.epoch, 2)
        self.assertEqual(enabled.global_step, 6)
        self.assertEqual(schedule.state['joint_train_batches_seen'], 3)
        self.assertEqual(schedule.context()['progress'], 3 / 8)
        # A contradictory legacy counter must not silently restart q at zero.
        broken = self.make_workspace(self.root / 'bad-legacy')
        payload['pickles']['global_step'] = dill.dumps(100)
        broken.load_payload(payload, include_keys=['epoch', 'global_step'])
        with self.assertRaises(ValueError):
            broken.prepare_gap_schedule(broken.cfg, [5, 5, 3])

    def test_new_baseline_checkpoint_rejects_changed_budget_when_enabling(self):
        self.cfg.training.progressive_gap_weighting.enabled = False
        old = self.make_workspace(self.root / 'baseline', stop_epoch=0)
        with self.assertRaises(StopAfterCheckpoint):
            old.run()
        with old.get_checkpoint_path().open('rb') as stream:
            payload = torch.load(stream, pickle_module=dill, weights_only=False)
        self.cfg.training.progressive_gap_weighting.enabled = True
        self.cfg.training.num_epochs = 5
        changed = self.make_workspace(self.root / 'changed')
        changed.load_payload(payload, include_keys=['epoch', 'global_step'])
        with self.assertRaisesRegex(ValueError, 'num_epochs'):
            changed.prepare_gap_schedule(changed.cfg, [5, 5])


if __name__ == '__main__':
    unittest.main()
