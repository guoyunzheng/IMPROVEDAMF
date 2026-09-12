import torch
import torchvision


def get_resnet(name, weights=None, **kwargs):
    """Build a torchvision ResNet observation encoder."""
    model = getattr(torchvision.models, name)(weights=weights, **kwargs)
    model.fc = torch.nn.Identity()
    return model
