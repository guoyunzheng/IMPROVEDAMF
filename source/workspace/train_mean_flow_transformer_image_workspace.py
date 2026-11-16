import os
from typing import Union
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
from source.workspace.base_workspace import BaseWorkspace
from source.policy.mean_flow_transformer_image_policy import MeanFlowTransformerImagePolicy
from source.policy.adaptive_mean_flow_transformer_image_policy import AdaptiveMeanFlowTransformerImagePolicy
from source.dataset.base_dataset import BaseImageDataset
from source.env_runner.base_image_runner import BaseImageRunner
from source.common.checkpoint_util import TopKCheckpointManager
from source.common.json_logger import JsonLogger
from source.common.mean_flow_schedule import scheduled_flow_ratio
from source.common.kinematic_mean_flow_schedule import set_kmf_progress
from source.common.progressive_gap import capture_rng, epoch_batch_sizes, restore_rng
from source.common.pytorch_util import dict_apply, optimizer_to
from source.model.diffusion.ema_model import EMAModel
from source.model.common.lr_scheduler import get_scheduler
from source.model.common.normalizer import SingleFieldLinearNormalizer

OmegaConf.register_new_resolver("eval", eval, replace=True)

class TrainMeanFlowTransformerImageWorkspace(BaseWorkspace):
    include_keys = ['global_step', 'epoch', 'kmf_resume_state']

    def __init__(self, cfg: OmegaConf, output_dir=None):
        super().__init__(cfg, output_dir=output_dir)
        if cfg.training.get('progressive_gap_weighting', {}).get('enabled', False):
            raise ValueError("Progressive gap weighting currently supports AMF lowdim only")

        # set seed
        seed = cfg.training.seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        # configure model
        self.model: MeanFlowTransformerImagePolicy = hydra.utils.instantiate(cfg.policy)

        self.ema_model: MeanFlowTransformerImagePolicy = None
        if cfg.training.use_ema:
            self.ema_model = copy.deepcopy(self.model)

        # configure training state
        self.enable_amp = str(cfg.training.device).startswith('cuda') and torch.cuda.is_available()
        self.optimizer = self.model.get_optimizer(**cfg.optimizer)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.enable_amp)

        # configure training state
        self.global_step = 0
        self.epoch = 0
        self.kmf_resume_state = None

    def load_payload(self, payload, exclude_keys=None, include_keys=None, **kwargs):
        if include_keys is not None:
            include_keys = list(dict.fromkeys([*include_keys, 'kmf_resume_state']))
        super().load_payload(payload, exclude_keys, include_keys, **kwargs)
        if self.kmf_resume_state is not None:
            self.epoch = self.kmf_resume_state['next_epoch']
            self.global_step = self.kmf_resume_state['next_global_step']

    def record_kmf_resume_state(self, lr_scheduler):
        if getattr(self.model, 'mf_target_type', 'meanflow') == 'kmf_learnable':
            self.kmf_resume_state = dict(
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

        # configure dataset
        dataset: BaseImageDataset
        dataset = hydra.utils.instantiate(cfg.task.dataset)
        assert isinstance(dataset, BaseImageDataset)
        val_dataset = dataset.get_validation_dataset()
        normalizer = dataset.get_normalizer()

        train_dataloader = DataLoader(dataset, **cfg.dataloader)
        val_dataloader = DataLoader(val_dataset, **cfg.val_dataloader)

        if isinstance(self.model, AdaptiveMeanFlowTransformerImagePolicy):
            if hasattr(dataset, 'action_key'):
                ac_key = getattr(dataset, 'action_key')
            else:
                ac_key = 'action'
            # Change action normalization to use standardization
            normalizer['action'] = SingleFieldLinearNormalizer.create_fit(data=dataset.replay_buffer[ac_key], mode='gaussian')

        # If using frozen image encoder, no need to save image encoder params
        exclude_obs_encoder = cfg.freeze_obs_encoder

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
        if self.kmf_resume_state is not None:
            lr_scheduler.load_state_dict(self.kmf_resume_state['lr_scheduler'])
            for group, lr in zip(self.optimizer.param_groups, lr_scheduler.get_last_lr()):
                group['lr'] = lr

        # configure ema
        ema: EMAModel = None
        if cfg.training.use_ema:
            ema = hydra.utils.instantiate(
                cfg.ema,
                model=self.ema_model)
            if getattr(self.model, 'mf_target_type', 'meanflow') == 'kmf_learnable':
                ema.optimization_step = self.global_step

        # configure env
        env_runner: BaseImageRunner
        env_runner = hydra.utils.instantiate(
            cfg.task.env_runner,
            output_dir=self.output_dir)
        assert isinstance(env_runner, BaseImageRunner)

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

        if cfg.training.debug:
            cfg.training.num_epochs = 2
            if getattr(self.model, 'mf_target_type', 'meanflow') == 'kmf_learnable':
                flow_schedule = cfg.training.get('flow_ratio_schedule')
                if flow_schedule is not None and flow_schedule.get('fm_only_epochs') is not None:
                    flow_schedule.fm_only_epochs = min(flow_schedule.fm_only_epochs, cfg.training.num_epochs)
            cfg.training.max_train_steps = 3
            cfg.training.max_val_steps = 3
            cfg.training.rollout_every = 1
            cfg.training.checkpoint_every = 1
            cfg.training.val_every = 1
            cfg.training.sample_every = 1

        batch_sizes = epoch_batch_sizes(len(dataset), cfg.dataloader.batch_size,
                                       cfg.dataloader.get('drop_last', False),
                                       cfg.training.max_train_steps)
        # Older image checkpoints count the full loader. Learned KMF checkpoints
        # record the actual capped batches so resuming keeps the gamma schedule.
        last_training_batch = len(train_dataloader) - 1
        if getattr(self.model, 'mf_target_type', 'meanflow') == 'kmf_learnable':
            last_training_batch = len(batch_sizes) - 1

        # JVP computation is not supported by flash attention or efficient attention kernels
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)

        if self.kmf_resume_state is not None:
            restore_rng(self.kmf_resume_state['rng'])

        # training loop
        log_path = os.path.join(self.output_dir, 'logs.json.txt')
        with JsonLogger(log_path) as json_logger:
            for local_epoch_idx in range(self.epoch, cfg.training.num_epochs):
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
                        kmf_details = set_kmf_progress(
                            self.model, self.ema_model, self.epoch, batch_idx,
                            cfg.training.num_epochs, len(batch_sizes),
                            cfg.training.get('flow_ratio_schedule'))
                        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=self.enable_amp):
                            raw_loss = self.model.compute_loss(batch)
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
                        step_log.update(kmf_details)
                        step_log.update(getattr(self.model, 'kmf_diagnostics', {}))

                        is_last_batch = (batch_idx == last_training_batch)
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
                rollout_this_epoch = False
                if ("last_n_rollout_every" in cfg.training) and (self.epoch >= (cfg.training.num_epochs - cfg.training.last_n_rollout_every.n)):
                    if (self.epoch % cfg.training.last_n_rollout_every.interval) == 0:
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
                if (self.epoch % cfg.training.val_every) == 0:
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
                        obs_dict = batch['obs']
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
                self.record_kmf_resume_state(lr_scheduler)
                if ((self.epoch % cfg.training.checkpoint_every) == 0) and rollout_this_epoch:
                    # checkpointing
                    if cfg.checkpoint.save_last_ckpt:
                        self.save_checkpoint(exclude_obs_encoder=exclude_obs_encoder)
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
                        self.save_checkpoint(path=topk_ckpt_path, exclude_obs_encoder=exclude_obs_encoder)
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
    workspace = TrainDiffusionTransformerHybridWorkspace(cfg)
    workspace.run()
