from typing import Dict, Tuple, Union
import copy
import torch
import torch.nn as nn
import torchvision
from source.model.vision.crop_randomizer import CropRandomizer
from source.model.common.module_attr_mixin import ModuleAttrMixin
from source.common.pytorch_util import dict_apply, replace_submodules
from source.model.vision.spatial_softmax import replace_avgpool_with_spatial_softmax
from torchvision.ops.misc import FrozenBatchNorm2d

class MultiImageObsEncoder(ModuleAttrMixin):
    def __init__(self,
            shape_meta: dict,
            rgb_model: Union[nn.Module, Dict[str,nn.Module]],
            resize_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            crop_shape: Union[Tuple[int,int], Dict[str,tuple], None]=None,
            random_crop: bool=True,
            # replace BatchNorm with GroupNorm
            use_group_norm: bool=False,
            # use single rgb model for all rgb inputs
            share_rgb_model: bool=False,
            frozen: bool=False,
            freeze_batchnorm: bool=False,
            imagenet_norm: bool=False,
            use_spatial_softmax: bool = False,
            eval_fixed_crop: bool = True,
            n_spatial_softmax_keypoints: int = 16,
            n_obs_steps:int =None
        ):
        """
        Assumes rgb input: B,C,H,W
        Assumes low_dim input: B,D
        """
        super().__init__()

        rgb_keys = list()
        low_dim_keys = list()
        key_model_map = nn.ModuleDict()
        key_transform_map = nn.ModuleDict()
        key_transform_map_eval = nn.ModuleDict()
        key_shape_map = dict()

        # handle sharing vision backbone
        if share_rgb_model:
            assert isinstance(rgb_model, nn.Module)
            if use_group_norm:
                rgb_model = replace_submodules(
                    root_module=rgb_model,
                    predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                    func=lambda x: nn.GroupNorm(
                        num_groups=x.num_features//16,
                        num_channels=x.num_features)
                )
            elif freeze_batchnorm:
                this_model = replace_batchnorm_with_frozen_batchnorm(rgb_model)

            if use_spatial_softmax:
                # We need shape = [C, crop_h, crop_w] here
                model_input_shape = [3] + crop_shape # Hard coded 3 for channel shape of rgb
                this_model = replace_avgpool_with_spatial_softmax(this_model, model_input_shape, num_kps=n_spatial_softmax_keypoints)
            key_model_map['rgb'] = rgb_model

        obs_shape_meta = shape_meta['obs']
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            type = attr.get('type', 'low_dim')
            key_shape_map[key] = shape
            if type == 'rgb':
                rgb_keys.append(key)
                # configure model for this key
                this_model = None
                if not share_rgb_model:
                    if isinstance(rgb_model, dict):
                        # have provided model for each key
                        this_model = rgb_model[key]
                    else:
                        assert isinstance(rgb_model, nn.Module)
                        # have a copy of the rgb model
                        this_model = copy.deepcopy(rgb_model)

                if this_model is not None:
                    if use_group_norm:
                        this_model = replace_submodules(
                            root_module=this_model,
                            predicate=lambda x: isinstance(x, nn.BatchNorm2d),
                            func=lambda x: nn.GroupNorm(
                                num_groups=x.num_features//16,
                                num_channels=x.num_features)
                        )
                    elif freeze_batchnorm:
                        this_model = replace_batchnorm_with_frozen_batchnorm(this_model)

                    if use_spatial_softmax:
                        # We need shape = [C, crop_h, crop_w] here
                        model_input_shape = [shape[0]] + crop_shape
                        this_model = replace_avgpool_with_spatial_softmax(this_model, model_input_shape, num_kps=n_spatial_softmax_keypoints)
                    key_model_map[key] = this_model

                # configure resize
                input_shape = shape
                this_resizer = nn.Identity()
                if resize_shape is not None:
                    if isinstance(resize_shape, dict):
                        h, w = resize_shape[key]
                    else:
                        h, w = resize_shape
                    this_resizer = torchvision.transforms.Resize(
                        size=(h,w)
                    )
                    input_shape = (shape[0],h,w)

                # configure randomizer
                this_randomizer = nn.Identity()
                if crop_shape is not None:
                    if isinstance(crop_shape, dict):
                        h, w = crop_shape[key]
                    else:
                        h, w = crop_shape
                    if random_crop:
                        this_randomizer = CropRandomizer(
                            input_shape=input_shape,
                            crop_height=h,
                            crop_width=w,
                            num_crops=1,
                            pos_enc=False,
                            n_obs_steps=n_obs_steps
                        )
                    else:
                        this_randomizer = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                # configure normalizer
                this_normalizer = nn.Identity()
                if imagenet_norm:
                    this_normalizer = torchvision.transforms.Normalize(
                        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

                this_transform = nn.Sequential(this_resizer, this_randomizer, this_normalizer)
                if random_crop and eval_fixed_crop:
                    randomizer_eval = torchvision.transforms.CenterCrop(
                            size=(h,w)
                        )
                    this_transform_eval = nn.Sequential(
                        this_resizer, randomizer_eval, this_normalizer)
                else:
                    this_transform_eval = this_transform
                key_transform_map[key] = this_transform
                key_transform_map_eval[key] = this_transform_eval
            elif type == 'low_dim':
                low_dim_keys.append(key)
            else:
                raise RuntimeError(f"Unsupported obs type: {type}")
        rgb_keys = sorted(rgb_keys)
        low_dim_keys = sorted(low_dim_keys)

        self.shape_meta = shape_meta
        self.key_model_map = key_model_map
        self.key_transform_map = key_transform_map
        self.key_transform_map_eval = key_transform_map_eval
        self.share_rgb_model = share_rgb_model
        self.rgb_keys = rgb_keys
        self.low_dim_keys = low_dim_keys
        self.key_shape_map = key_shape_map

        if frozen:
            print("Using frozen vision encoder!")
            for model in self.key_model_map.values():
                model.requires_grad_(False)

        self.frozen = frozen

    def forward(self, obs_dict):
        batch_size = None
        features = list()
        # process rgb input
        if self.share_rgb_model:
            # pass all rgb obs to rgb model
            imgs = list()
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                if self.training:
                    img = self.key_transform_map[key](img)
                else:
                    img = self.key_transform_map_eval[key](img)
                imgs.append(img)
            # (N*B,C,H,W)
            imgs = torch.cat(imgs, dim=0)
            # (N*B,D)
            feature = self.key_model_map['rgb'](imgs)
            # (N,B,D)
            feature = feature.reshape(-1,batch_size,*feature.shape[1:])
            # (B,N,D)
            feature = torch.moveaxis(feature,0,1)
            # (B,N*D)
            feature = feature.reshape(batch_size,-1)
            features.append(feature)
        else:
            # run each rgb obs to independent model
            for key in self.rgb_keys:
                img = obs_dict[key]
                if batch_size is None:
                    batch_size = img.shape[0]
                else:
                    assert batch_size == img.shape[0]
                assert img.shape[1:] == self.key_shape_map[key]
                if self.training:
                    img = self.key_transform_map[key](img)
                else:
                    img = self.key_transform_map_eval[key](img)
                feature = self.key_model_map[key](img)
                features.append(feature)

        # process lowdim input
        for key in self.low_dim_keys:
            data = obs_dict[key]
            if batch_size is None:
                batch_size = data.shape[0]
            else:
                assert batch_size == data.shape[0]
            assert data.shape[1:] == self.key_shape_map[key]
            features.append(data)

        # concatenate all features
        result = torch.cat(features, dim=-1)
        return result

    def obs_encode(self, obs_dict):
        if self.frozen:
            with torch.no_grad():
                return self(obs_dict)
        else:
            return self(obs_dict)

    @torch.no_grad()
    def output_shape(self):
        example_obs_dict = dict()
        obs_shape_meta = self.shape_meta['obs']
        batch_size = 1
        for key, attr in obs_shape_meta.items():
            shape = tuple(attr['shape'])
            this_obs = torch.zeros(
                (batch_size,) + shape,
                dtype=self.dtype,
                device=self.device)
            example_obs_dict[key] = this_obs
        example_output = self.forward(example_obs_dict)
        output_shape = example_output.shape[1:]
        return output_shape


def replace_batchnorm_with_frozen_batchnorm(model: ModuleAttrMixin):
    for name, child in list(model.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            fbn = FrozenBatchNorm2d(child.num_features, eps=child.eps)
            with torch.no_grad():
                if child.affine:
                    fbn.weight.copy_(child.weight.data)
                    fbn.bias.copy_(child.bias.data)
                fbn.running_mean.copy_(child.running_mean.data)
                fbn.running_var.copy_(child.running_var.data)
            setattr(model, name, fbn)
        else:
            # recursively replace in submodules
            replace_batchnorm_with_frozen_batchnorm(child)
    return model


    # bn_list = [m for k,m
    #     in vision_enc.key_model_map['image'].named_modules()]
    # print(bn_list)
