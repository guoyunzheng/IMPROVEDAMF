import os
import hydra
import torch
from omegaconf import OmegaConf
import pathlib
from torch.utils.data import DataLoader
import copy
import random
import wandb
import tqdm
import numpy as np
import shutil
from typing import Union
from source.workspace.base_workspace import BaseWorkspace
from source.policy.mean_flow_transformer_lowdim_policy import MeanFlowTransformerLowdimPolicy
from source.policy.adaptive_mean_flow_transformer_lowdim_policy import AdaptiveMeanFlowTransformerLowdimPolicy
from source.dataset.base_dataset import BaseLowdimDataset
from source.env_runner.base_lowdim_runner import BaseLowdimRunner
from source.common.checkpoint_util import TopKCheckpointManager
from source.common.json_logger import JsonLogger
from source.common.mean_flow_schedule import scheduled_flow_ratio
from source.common.progressive_gap import (
    GapWeightingSchedule, gap_config, epoch_batch_sizes, capture_rng, restore_rng,
    isolated_rng, aggregate_diagnostics,
)
from source.common.pytorch_util import dict_apply, optimizer_to
from source.model.diffusion.ema_model import EMAModel
from source.model.common.lr_scheduler import get_scheduler
from source.model.common.normalizer import SingleFieldLinearNormalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainMeanFlowLowdimWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'progressive_gap_state', 'training_resume_state']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: MeanFlowTransformerLowdimPolicy = hydra.utils.instantiate(cfg.policy)
        self.gap_cfg = gap_config(cfg.training.get('progressive_gap_weighting'))
        if self.gap_cfg['enabled'] and not isinstance(self.model, AdaptiveMeanFlowTransformerLowdimPolicy):
            raise ValueError("Progressive gap weighting currently supports AMF lowdim only")

        self.ema_model: MeanFlowTransformerLowdimPolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.enable_amp = str(cfg.training.device).startswith('cuda') and torch.cuda.is_available()

        # Transformer
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        # JVP computation is not supported by flash attention or efficient attention kernels
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_amp)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.progressive_gap_state = None
        self.training_resume_state = None
        self._legacy_checkpoint_progress = None
        self._loaded_training_cfg = None

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        # New checkpoints explicitly record the NEXT training position. The
        # visible epoch/global_step still identify the completed evaluation row.
        if include_keys is not None:
            include_keys = list(dict.fromkeys([*include_keys, 'progressive_gap_state', 'training_resume_state']))
        super().load_payload(payload, exclude_keys, include_keys, **kwargs)
        self._loaded_training_cfg = payload['cfg']
        if self.training_resume_state is not None:
            self.epoch = self.training_resume_state['next_epoch']
            self.global_step = self.training_resume_state['next_global_step']
        else:
            self._legacy_checkpoint_progress = (self.epoch, self.global_step, payload['cfg'])

    def prepare_gap_schedule(self, cfg, batch_sizes):
        if self.gap_cfg['enabled'] and self._loaded_training_cfg is not None and self.progressive_gap_state is None:
            # Also check new baseline checkpoints when enabling the curriculum
            # for the first time. Completed batches must be verifiable.
            old_cfg = self._loaded_training_cfg
            for key in ('num_epochs', 'max_train_steps', 'debug', 'flow_ratio_schedule'):
                if old_cfg.training.get(key) != self.cfg.training.get(key):
                    raise ValueError(f"Cannot infer gap progress after changing training.{key}")
            for key in ('horizon', 'n_obs_steps', 'monotone_schedule_ratio', 'time_dist', 'flow_ratio'):
                if old_cfg.policy.get(key) != self.cfg.policy.get(key):
                    raise ValueError(f"Cannot infer gap calibration after changing policy.{key}")
            for key in ('batch_size', 'drop_last'):
                if old_cfg.dataloader.get(key) != cfg.dataloader.get(key):
                    raise ValueError(f"Cannot infer consumed batches after changing dataloader.{key}")
        if self._legacy_checkpoint_progress is not None and self.gap_cfg['enabled']:
            epoch, step, old_cfg = self._legacy_checkpoint_progress
            # Legacy saves occur after completing an epoch, before both counters
            # advance. Only infer progress when the saved budget/step count proves it.
            if (old_cfg.training.debug or old_cfg.training.max_train_steps is not None
                    or old_cfg.policy.flow_ratio != cfg.policy.flow_ratio
                    or (epoch + 1) * len(batch_sizes) != step + 1):
                raise ValueError("Cannot verify legacy consumed batches for gap weighting")
            self.epoch, self.global_step = epoch + 1, step + 1
        if (self.gap_cfg['enabled'] and self.training_resume_state is not None
                and self.global_step != self.epoch * len(batch_sizes)):
            raise ValueError("Saved next epoch and consumed training batches disagree")
        if not self.gap_cfg['enabled']:
            if self.progressive_gap_state is not None and self.progressive_gap_state['config']['enabled']:
                raise ValueError("Cannot silently disable a checkpoint's gap curriculum on resume")
            return None
        schedule = GapWeightingSchedule(
            self.gap_cfg, self.model, cfg.training.num_epochs, batch_sizes,
            cfg.policy.flow_ratio, cfg.training.get('flow_ratio_schedule'),
            self.epoch, self.progressive_gap_state)
        self.progressive_gap_state = schedule.state
        return schedule

    def record_resume_state(self, lr_scheduler):
        self.training_resume_state = dict(
            next_epoch=self.epoch + 1, next_global_step=self.global_step + 1,
            rng=capture_rng(), lr_scheduler=lr_scheduler.state_dict())

    def run(self):
        cfg = copy.deepcopy(self.cfg)

        # resume training
        if cfg.training.resume:
            lastest_ckpt_path = self.get_checkpoint_path()
            if lastest_ckpt_path.is_file():
                print(f"Resuming from checkpoint {lastest_ckpt_path}")
                self.load_checkpoint(path=lastest_ckpt_path)

        # Resolve debug limits before constructing either schedule.
        if cfg.training.debug:
            cfg.training.num_epochs = 2
            flow_schedule = cfg.training.get('flow_ratio_schedule')
            if flow_schedule is not None and flow_schedule.get('fm_only_epochs') is not None:
                flow_schedule.fm_only_epochs = min(flow_schedule.fm_only_epochs, cfg.training.num_epochs)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        # configure dataset
        dataset: BaseLowdimDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseLowdimDataset)
        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        batch_sizes = epoch_batch_sizes(len(dataset), cfg.dataloader.batch_size,
                                       cfg.dataloader.get('drop_last', False),
                                       cfg.training.max_train_steps)
        gap_schedule = self.prepare_gap_schedule(cfg, batch_sizes)
        normalizer = dataset.get_normalizer()

        # configure validation dataset
        val_dataset = dataset.get_validation_dataset()
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if isinstance(self.model, AdaptiveMeanFlowTransformerLowdimPolicy):
            # Change action normalization to use standardization
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(data=dataset.replay_buffer['action'], mode='gaussian')

        self.model.set_normalizer(normalizer)
        if cfg.training.use_ema:
            self.ema_model.set_normalizer(normalizer)

        # configure lr scheduler
        lr_scheduler = get_scheduler(
            cfg.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=cfg.training.lr_warmup_steps,
            num_training_steps=(
                len(train_dataloader) * cfg.training.num_epochs) \
                    // cfg.training.gradient_accumulate_every,
            # pytorch assumes stepping LRScheduler every epoch
            # however huggingface diffusers steps it every batch
            last_epoch=self.global_step-1
        )
        if self.training_resume_state is not None:
            lr_scheduler.load_state_dict(self.training_resume_state['lr_scheduler'])
            for group, lr in zip(self.optimizer.param_groups, lr_scheduler.get_last_lr()):
                group['lr'] = lr

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            # EMA is updated once per training batch. Its warmup counter is
            # not stored in legacy checkpoints; recover it from batch progress.
            ema.optimization_step = self.global_step

        # configure env
        env_runner: BaseLowdimRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseLowdimRunner)

        # configure logging
        wandb_run = wandb.init(
            dir=str(self.output_dir),
            config=OmegaConf.to_container(cfg, resolve=True),
            **cfg.logging
        )
        wandb.config.update(
            {
                "output_dir": self.output_dir,
            }
        )
        if gap_schedule is not None:
            # Persist resolved lambda, budget and sampling parameters alongside
            # Hydra's input config; also included in every checkpoint.
            OmegaConf.save(OmegaConf.create(self.progressive_gap_state),
                           os.path.join(self.output_dir, 'progressive_gap_config.yaml'))
            wandb.config.update({'progressive_gap_resolved': self.progressive_gap_state})

        # configure checkpoint
        topk_manager = TopKCheckpointManager(
            save_dir=os.path.join(self.output_dir, 'checkpoints'),
            **cfg.checkpoint.topk
        )

        # device transfer
        device = torch.device(cfg.training.device)
        self.model.to(device)
        if self.ema_model is not None:
            self.ema_model.to(device)
        optimizer_to(self.optimizer, device)

        # save batch for sampling
        train_sampling_batch = None

        if self.training_resume_state is not None:
            # Setup and calibration cannot consume the restored training RNG.
            restore_rng(self.training_resume_state['rng'])

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
                is_final_epoch = self.epoch == cfg.training.num_epochs - 1
                # Use the restored absolute epoch, also for validation and EMA.
                flow_ratio = scheduled_flow_ratio(
                    self.epoch, cfg.training.num_epochs, cfg.policy.flow_ratio,
                    cfg.training.get('flow_ratio_schedule'))
                self.model.flow_ratio = flow_ratio
                if self.ema_model is not None:
                    self.ema_model.flow_ratio = flow_ratio
                step_log = dict()
                # ========= train for this epoch ==========
                train_losses = list()
                with tqdm.tqdm(train_dataloader, desc=f"Training epoch {self.epoch}",
                        leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                    for batch_idx, batch in enumerate(tepoch):
                        # device transfer
                        batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                        if train_sampling_batch is None:
                            train_sampling_batch = batch

                        # compute loss
                        gap_details = {}
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.enable_amp):
                            if gap_schedule is None:
                                raw_loss = self.model.compute_loss(batch)
                            else:
                                context = gap_schedule.context()
                                raw_loss, gap_details = self.model.compute_loss(
                                    batch, gap_context=context, return_diagnostics=True)
                                gap_details['joint_train_batches_seen'] = self.progressive_gap_state['joint_train_batches_seen']
                                if context is not None:
                                    gap_details.update(gap_progress=context['progress'], gap_lambda=context['gap_lambda'])
                            loss = raw_loss / cfg.training.gradient_accumulate_every

                        self.scaler.scale(loss).backward()

                        # step optimizer
                        if self.global_step % cfg.training.gradient_accumulate_every == 0:
                            self.scaler.step(self.optimizer)
                            self.scaler.update()
                            self.optimizer.zero_grad()
                            lr_scheduler.step()

                        # update ema
                        if cfg.training.use_ema:
                            ema.step(self.model)
                        if gap_schedule is not None:
                            gap_schedule.consume(self.epoch)

                        # logging
                        raw_loss_cpu = raw_loss.item()
                        tepoch.set_postfix(loss=raw_loss_cpu, refresh=False)
                        train_losses.append(raw_loss_cpu)
                        step_log = {
                            'train_loss': raw_loss_cpu,
                            'global_step': self.global_step,
                            'epoch': self.epoch,
                            'fm_ratio': flow_ratio,
                            'mf_ratio': 1.0 - flow_ratio,
                            'lr': lr_scheduler.get_last_lr()[0]
                        }
                        step_log.update(gap_details)

                        is_last_batch = (batch_idx == len(batch_sizes) - 1)
                        if not is_last_batch:
                            # log of last step is combined with validation and rollout
                            wandb_run.log(step_log, step=self.global_step)
                            json_logger.log(step_log)
                            self.global_step += 1

                        if (cfg.training.max_train_steps is not None) \
                            and batch_idx >= (cfg.training.max_train_steps-1):
                            break

                # at the end of each epoch
                # replace train_loss with epoch average
                train_loss = np.mean(train_losses)
                step_log['train_loss'] = train_loss

                # ========= eval for this epoch ==========
                policy = self.model
                if cfg.training.use_ema:
                    policy = self.ema_model
                policy.eval()

                # run rollout
                rollout_this_epoch = is_final_epoch
                in_final_rollout_window = (
                    "last_n_rollout_every" in cfg.training
                    and self.epoch >= cfg.training.num_epochs - cfg.training.last_n_rollout_every.n)
                if in_final_rollout_window:
                    # Align to the end, including totals not divisible by 10.
                    epochs_until_end = cfg.training.num_epochs - 1 - self.epoch
                    if (epochs_until_end % cfg.training.last_n_rollout_every.interval) == 0:
                        rollout_this_epoch = True

                elif (self.epoch % cfg.training.rollout_every) == 0:
                    rollout_this_epoch = True

                if rollout_this_epoch:
                    if cfg.inference_sweep.perform_inference_sweep:
                        runner_log = env_runner.sweep_run(policy, cfg.inference_sweep.sweep_param_dict)
                    else:
                        runner_log = env_runner.run(policy)

                    if hasattr(policy, 'reset_history'):
                        policy.reset_history()
                    # log all
                    step_log.update(runner_log)

                # run validation
                if (self.epoch % cfg.training.val_every) == 0 or is_final_epoch:
                    if gap_schedule is not None:
                        # Fixed, unweighted FM and MF protocols across epochs.
                        # Extra evaluation forwards never advance the curriculum.
                        was_training = self.model.training
                        self.model.eval()
                        rows = []
                        try:
                            with isolated_rng(self.gap_cfg['validation_seed']), torch.no_grad():
                                for batch_idx, batch in enumerate(val_dataloader):
                                    batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))
                                    self.model.flow_ratio = 1.0
                                    _, diagonal = self.model.compute_loss(batch, return_diagnostics=True)
                                    rows.append(diagonal)
                                    if self.progressive_gap_state['active']:
                                        self.model.flow_ratio = 0.0
                                        _, intervals = self.model.compute_loss(batch, return_diagnostics=True)
                                        rows.append(intervals)
                                    if cfg.training.max_val_steps is not None and batch_idx >= cfg.training.max_val_steps - 1:
                                        break
                        finally:
                            self.model.flow_ratio = flow_ratio
                            self.model.train(was_training)
                        step_log.update({'val_' + key: value for key, value in aggregate_diagnostics(rows).items()})
                    # Retain val_loss for existing result readers. With the
                    # curriculum enabled this remains an UNWEIGHTED reference.
                    with torch.no_grad():
                        val_losses = list()
                        with tqdm.tqdm(val_dataloader, desc=f"Validation epoch {self.epoch}",
                                leave=False, mininterval=cfg.training.tqdm_interval_sec) as tepoch:
                            for batch_idx, batch in enumerate(tepoch):
                                batch = dict_apply(batch, lambda x: x.to(device, non_blocking=True))

                                loss = self.model.compute_loss(batch)
                                val_losses.append(loss)
                                if (cfg.training.max_val_steps is not None) \
                                    and batch_idx >= (cfg.training.max_val_steps-1):
                                    break
                        if len(val_losses) > 0:
                            val_loss = torch.mean(torch.tensor(val_losses)).item()
                            # log epoch average validation loss
                            step_log['val_loss'] = val_loss

                # run diffusion sampling on a training batch
                if (self.epoch % cfg.training.sample_every) == 0:
                    with torch.no_grad():
                        # sample trajectory from training set, and evaluate difference
                        batch = dict_apply(train_sampling_batch, lambda x: x.to(device, non_blocking=True))
                        obs_dict = {'obs': batch['obs']}
                        gt_action = batch['action']
                        result = policy.predict_action(obs_dict)
                        if hasattr(policy, 'reset_history'):
                            policy.reset_history()
                        pred_action = result['action_pred']
                        mse = torch.nn.functional.mse_loss(pred_action, gt_action)
                        step_log['train_action_mse_error'] = mse.item()
                        del batch
                        del obs_dict
                        del gt_action
                        del result
                        del pred_action
                        del mse

                # checkpoint
                # Keep densely evaluated final checkpoints eligible for top-k,
                # including the last three evaluations between regular saves.
                if ((self.epoch % cfg.training.checkpoint_every) == 0 or is_final_epoch or in_final_rollout_window) and rollout_this_epoch:
                    self.record_resume_state(lr_scheduler)
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint()
                    if cfg.checkpoint.save_last_snapshot:
                        self.save_snapshot()

                    # sanitize metric names
                    metric_dict = dict()
                    for key, value in step_log.items():
                        new_key = key.replace('/', '_')
                        metric_dict[new_key] = value

                    # We can't copy the last checkpoint here
                    # since save_checkpoint uses threads.
                    # therefore at this point the file might have been empty!
                    topk_ckpt_path = topk_manager.get_ckpt_path(metric_dict)

                    if topk_ckpt_path is not None:
                        self.save_checkpoint(path=topk_ckpt_path)
                # ========= eval end for this epoch ==========
                policy.train()

                # end of epoch
                # log of last step is combined with validation and rollout
                wandb_run.log(step_log, step=self.global_step)
                json_logger.log(step_log)
                self.global_step += 1
                self.epoch += 1

@hydra.main(
    version_base=None,
    config_path=str(pathlib.Path(__file__).parent.parent.joinpath("config")),
    config_name=pathlib.Path(__file__).stem)
def main(cfg):
    workspace = TrainDiffusionUnetHybridWorkspace(cfg)
    workspace.run()
