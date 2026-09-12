from typing import Dict
import torch
import torch.nn as nn
from source.model.common.module_attr_mixin import ModuleAttrMixin
from source.model.common.normalizer import LinearNormalizer
from source.model.vision.multi_image_obs_encoder import MultiImageObsEncoder
from source.common.pytorch_util import dict_apply

class BaseImagePolicy(ModuleAttrMixin):
    def __init__(self,
                 obs_encoder: MultiImageObsEncoder,
                 n_obs_steps: int,
                 freeze_obs_encoder: bool = False,):
        super().__init__()

        self.obs_encoder = obs_encoder # Observation encoder module
        self.n_obs_steps = n_obs_steps

        if freeze_obs_encoder:
            obs_encoder.requires_grad_(False)
        self.freeze_obs_encoder = freeze_obs_encoder
        self.normalizer = LinearNormalizer()

    def process_obs(self, obs_dict):

        To = self.n_obs_steps
        nobs = self.normalizer.normalize(obs_dict)
        this_nobs = dict_apply(nobs, lambda x: x[:,:To,...].reshape(-1,*x.shape[2:]))
        if self.freeze_obs_encoder:
            with torch.no_grad():
                nobs_features = self.obs_encoder(this_nobs)
        else:
            nobs_features = self.obs_encoder(this_nobs)
        return nobs_features


    def predict_action(self, obs_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        obs_dict:
            str: B,To,*
        return: B,Ta,Da
        """
        raise NotImplementedError()

    # reset state for stateful policies
    def reset(self):
        pass

    # ========== training ===========
    # no standard training interface except setting normalizer
    def set_normalizer(self, normalizer: LinearNormalizer):
        self.normalizer.load_state_dict(normalizer.state_dict())
