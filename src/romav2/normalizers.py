import torch

_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
_INCEPTION_MEAN = torch.tensor([0.5, 0.5, 0.5])[None, :, None, None]
_INCEPTION_STD = torch.tensor([0.5, 0.5, 0.5])[None, :, None, None]


def imagenet(img: torch.Tensor) -> torch.Tensor:
    mean = _IMAGENET_MEAN.to(device=img.device)
    std = _IMAGENET_STD.to(device=img.device)
    return (img - mean) / std


def inception(img: torch.Tensor) -> torch.Tensor:
    mean = _INCEPTION_MEAN.to(device=img.device)
    std = _INCEPTION_STD.to(device=img.device)
    return (img - mean) / std
