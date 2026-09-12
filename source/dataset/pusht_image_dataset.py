from typing import Dict
import torch
import numpy as np
import copy
from source.common.pytorch_util import dict_apply
from source.common.replay_buffer import ReplayBuffer
from source.common.sampler import (
    SequenceSampler, get_val_mask, downsample_mask)
from source.model.common.normalizer import LinearNormalizer
from source.dataset.base_dataset import BaseImageDataset
from source.common.normalize_util import get_image_range_normalizer

class PushTImageDataset(BaseImageDataset):

    def __init__(self,
            zarr_path,
            abs_action=True,
            horizon=1,
            pad_before=0,
            pad_after=0,
            seed=42,
            val_ratio=0.0,
            max_train_episodes=None,
            n_obs_steps=None,
            ):

        super().__init__()

        self.replay_buffer = ReplayBuffer.copy_from_path(
            zarr_path, keys=['img', 'state', 'action'])
        val_mask = get_val_mask(
            n_episodes=self.replay_buffer.n_episodes,
            val_ratio=val_ratio,
            seed=seed)
        train_mask = ~val_mask
        train_mask = downsample_mask(
            mask=train_mask,
            max_n=max_train_episodes,
            seed=seed)

        # Only load relevant observations
        key_first_k = dict()
        if n_obs_steps is not None:
            key_first_k['img'] = n_obs_steps

        self.sampler_type = SequenceSampler

        self.sampler = self.sampler_type(
            replay_buffer=self.replay_buffer,
            sequence_length=horizon,
            pad_before=pad_before,
            pad_after=pad_after,
            episode_mask=train_mask,
            key_first_k=key_first_k)
        self.abs_action = abs_action
        self.train_mask = train_mask
        self.horizon = horizon
        self.pad_before = pad_before
        self.pad_after = pad_after
        self.key_first_k = key_first_k

    def get_validation_dataset(self):
        val_set = copy.copy(self)
        val_set.sampler = self.sampler_type(
            replay_buffer=self.replay_buffer,
            sequence_length=self.horizon,
            pad_before=self.pad_before,
            pad_after=self.pad_after,
            episode_mask=~self.train_mask,
            key_first_k=self.key_first_k
            )
        val_set.train_mask = ~self.train_mask
        return val_set


    def get_normalizer(self, mode='limits', **kwargs):
        data = {
            'action': self.replay_buffer['action'],
            'agent_pos': self.replay_buffer['state'][...,:2]
        }
        if not self.abs_action:
            data['action'] = data['action'] - data['agent_pos']
        normalizer = LinearNormalizer()
        normalizer.fit(data=data, last_n_dims=1, mode=mode, **kwargs)
        normalizer['image'] = get_image_range_normalizer()
        return normalizer

    def __len__(self) -> int:
        return len(self.sampler)

    def _sample_to_data(self, sample):

        agent_pos = sample['state'][:,:2].astype(np.float32) # (agent_posx2, block_posex3)
        image = np.moveaxis(sample['img'],-1,1)/255
        action = sample['action'].astype(np.float32)
        if not self.abs_action:
            action = action - agent_pos

        data = {
            'obs': {
                'image': image, # T, 3, 96, 96
                'agent_pos': agent_pos, # T, 2
            },
            'action': action # T, 2
        }

        return data


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.sampler.sample_sequence(idx)
        data = self._sample_to_data(sample)
        torch_data = dict_apply(data, torch.from_numpy)
        return torch_data
