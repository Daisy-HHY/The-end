from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from four_flower.models.registry import SUPPORTED_BACKBONES, create_backbone, create_model, list_backbones


def _forward_shape(model: nn.Module) -> torch.Size:
    model.eval()
    inputs = torch.randn(2, 3, 160, 160)
    with torch.no_grad():
        outputs = model(inputs)
    return outputs.shape


def test_list_backbones_matches_supported_constant():
    assert list_backbones() == SUPPORTED_BACKBONES
    assert SUPPORTED_BACKBONES == (
        "efficientnet_b0",
        "mobilenet_v3_small",
        "resnet18",
        "vit_tiny",
    )


@pytest.mark.parametrize("backbone_name", SUPPORTED_BACKBONES)
def test_create_backbone_forward_smoke_pass(backbone_name: str):
    model = create_backbone(backbone_name, num_classes=4, pretrained=False)
    assert isinstance(model, nn.Module)
    assert _forward_shape(model) == (2, 4)


def test_create_model_defaults_to_efficientnet_b0():
    model = create_model(num_classes=4, pretrained=False)
    assert _forward_shape(model) == (2, 4)


def test_create_backbone_accepts_legacy_use_pretrained_alias():
    model = create_backbone("efficientnet_b0", num_classes=4, use_pretrained=False)
    assert _forward_shape(model) == (2, 4)


def test_create_backbone_rejects_unknown_backbone():
    with pytest.raises(ValueError, match="Unsupported backbone 'unknown_backbone'"):
        create_backbone("unknown_backbone", num_classes=4, pretrained=False)


def test_create_backbone_rejects_unexpected_kwargs():
    with pytest.raises(TypeError, match="Unexpected keyword arguments: foo"):
        create_backbone("efficientnet_b0", num_classes=4, pretrained=False, foo=True)


def test_torchvision_backbone_dispatch_replaces_classifier(monkeypatch: pytest.MonkeyPatch):
    class FakeEfficientNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.classifier = nn.Sequential(nn.Dropout(), nn.Linear(8, 3))

        def forward(self, x):
            pooled = x.mean(dim=(2, 3))
            features = torch.cat([pooled, pooled, pooled[:, :2]], dim=1)
            return self.classifier(features)

    builder_calls = []

    def fake_builder(*, weights=None):
        builder_calls.append(weights)
        return FakeEfficientNet()

    fake_models = SimpleNamespace(
        efficientnet_b0=fake_builder,
        EfficientNet_B0_Weights=SimpleNamespace(DEFAULT="eff-default"),
    )

    monkeypatch.setattr("four_flower.models.registry.import_module", lambda name: fake_models)

    model = create_backbone("efficientnet_b0", num_classes=4, pretrained=True)
    assert builder_calls == ["eff-default"]
    assert model.classifier[-1].out_features == 4
    assert _forward_shape(model) == (2, 4)


def test_vit_tiny_dispatch_uses_timm_real_tiny_config(monkeypatch: pytest.MonkeyPatch):
    class FakeVitTiny(nn.Module):
        def forward(self, x):
            batch_size = x.shape[0]
            return torch.zeros(batch_size, 4)

    create_calls = []

    def fake_create_model(model_name: str, pretrained: bool, num_classes: int, img_size: int):
        create_calls.append(
            {
                "model_name": model_name,
                "pretrained": pretrained,
                "num_classes": num_classes,
                "img_size": img_size,
            }
        )
        return FakeVitTiny()

    fake_timm = SimpleNamespace(create_model=fake_create_model)

    monkeypatch.setattr("four_flower.models.registry.import_module", lambda name: fake_timm)

    model = create_backbone("vit_tiny", num_classes=4, pretrained=False)
    assert create_calls == [
        {
            "model_name": "vit_tiny_patch16_224",
            "pretrained": False,
            "num_classes": 4,
            "img_size": 160,
        }
    ]
    assert _forward_shape(model) == (2, 4)
