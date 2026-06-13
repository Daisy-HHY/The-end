from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.models import EfficientNet_B0_Weights, efficientnet_b0


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def build_transforms(image_size: int = 224, random_erasing_prob: float = 0.0) -> tuple[transforms.Compose, transforms.Compose]:
    train_steps = [
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ]
    if random_erasing_prob > 0:
        train_steps.append(
            transforms.RandomErasing(
                p=random_erasing_prob,
                scale=(0.02, 0.12),
                ratio=(0.3, 3.3),
                value="random",
            )
        )

    train_transform = transforms.Compose(
        train_steps
    )
    eval_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.14)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, eval_transform


def create_model(num_classes: int, use_pretrained: bool = True) -> torch.nn.Module:
    weights = None
    if use_pretrained:
        try:
            weights = EfficientNet_B0_Weights.DEFAULT
        except Exception:
            weights = None

    try:
        model = efficientnet_b0(weights=weights)
    except Exception as exc:
        print(f"Warning: failed to load pretrained EfficientNet-B0 weights, fallback to random init. {exc}")
        model = efficientnet_b0(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = torch.nn.Linear(in_features, num_classes)
    return model


def save_json(data: dict, output_path: Path) -> None:
    output_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_image(image_path: Path, image_size: int = 224) -> torch.Tensor:
    _, eval_transform = build_transforms(image_size=image_size)
    image = Image.open(image_path).convert("RGB")
    return eval_transform(image).unsqueeze(0)
