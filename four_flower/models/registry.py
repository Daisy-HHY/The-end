from __future__ import annotations

from importlib import import_module

from torch import nn


SUPPORTED_BACKBONES = (
    "efficientnet_b0",
    "mobilenet_v3_small",
    "resnet18",
    "vit_tiny",
)

_TORCHVISION_BACKBONES = {
    "efficientnet_b0": {
        "builder": "efficientnet_b0",
        "weights": "EfficientNet_B0_Weights",
        "classifier": "classifier",
    },
    "mobilenet_v3_small": {
        "builder": "mobilenet_v3_small",
        "weights": "MobileNet_V3_Small_Weights",
        "classifier": "classifier",
    },
    "resnet18": {
        "builder": "resnet18",
        "weights": "ResNet18_Weights",
        "classifier": "fc",
    },
}

_VIT_TINY_MODEL_NAME = "vit_tiny_patch16_224"
_VIT_TINY_IMAGE_SIZE = 160


def list_backbones() -> tuple[str, ...]:
    return SUPPORTED_BACKBONES


def create_backbone(backbone_name: str, num_classes: int, pretrained: bool = True, **kwargs) -> nn.Module:
    if "use_pretrained" in kwargs:
        pretrained = kwargs.pop("use_pretrained")
    if kwargs:
        unexpected = ", ".join(sorted(kwargs))
        raise TypeError(f"Unexpected keyword arguments: {unexpected}")

    if backbone_name not in SUPPORTED_BACKBONES:
        supported = ", ".join(SUPPORTED_BACKBONES)
        raise ValueError(f"Unsupported backbone '{backbone_name}'. Supported backbones: {supported}")

    if backbone_name == "vit_tiny":
        return _create_vit_tiny(num_classes=num_classes, pretrained=pretrained)

    return _create_torchvision_backbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def create_model(
    num_classes: int,
    use_pretrained: bool = True,
    backbone_name: str = "efficientnet_b0",
    pretrained: bool | None = None,
) -> nn.Module:
    if pretrained is None:
        pretrained = use_pretrained
    return create_backbone(
        backbone_name=backbone_name,
        num_classes=num_classes,
        pretrained=pretrained,
    )


def _create_torchvision_backbone(backbone_name: str, num_classes: int, pretrained: bool) -> nn.Module:
    spec = _TORCHVISION_BACKBONES[backbone_name]
    models = import_module("torchvision.models")
    builder = getattr(models, spec["builder"])
    model = builder(weights=_resolve_torchvision_weights(models, spec["weights"], pretrained))
    _replace_classifier(model, spec["classifier"], num_classes)
    return model


def _create_vit_tiny(num_classes: int, pretrained: bool) -> nn.Module:
    timm = import_module("timm")
    return timm.create_model(
        _VIT_TINY_MODEL_NAME,
        pretrained=pretrained,
        num_classes=num_classes,
        img_size=_VIT_TINY_IMAGE_SIZE,
    )


def _resolve_torchvision_weights(models, weights_name: str, pretrained: bool):
    if not pretrained:
        return None
    weights_enum = getattr(models, weights_name)
    return weights_enum.DEFAULT


def _replace_classifier(model: nn.Module, classifier_type: str, num_classes: int) -> None:
    if classifier_type == "classifier":
        in_features = model.classifier[-1].in_features
        model.classifier[-1] = nn.Linear(in_features, num_classes)
        return

    if classifier_type == "fc":
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        return

    raise TypeError(f"Unsupported classifier layout '{classifier_type}' for model type {type(model).__name__}.")
