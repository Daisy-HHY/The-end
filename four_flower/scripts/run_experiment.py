"""Unified experiment driver for backbone, ablation, and sensitivity studies."""

from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import torch
from sklearn.metrics import classification_report, confusion_matrix
from torch import nn
from torchvision import transforms
from torchvision.datasets import ImageFolder

from four_flower.efficientnet_utils import IMAGENET_MEAN, IMAGENET_STD, get_device, save_json, set_seed
from four_flower.models.registry import create_backbone, list_backbones
from four_flower.train_efficientnet import (
    TransformSubset,
    create_datasets,
    create_loaders,
    evaluate,
    run_epoch,
    save_confusion_matrix,
    save_training_plot,
)


PRESETS = {
    "eff_b0_full": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "resnet18_full": {
        "backbone": "resnet18",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "mobilenet_v3_small_full": {
        "backbone": "mobilenet_v3_small",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "vit_tiny_full": {
        "backbone": "vit_tiny",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "ablate_no_pretrain": {
        "backbone": "efficientnet_b0",
        "pretrained": False,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 0,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "ablate_no_freeze": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 0,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "ablate_no_ls": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.0,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "ablate_min_aug": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": False,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "hyper_h1_img128": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 128,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "hyper_h2_img192": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 192,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "hyper_h3_bs8": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 8,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "hyper_h4_bs32": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 32,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
    "hyper_h5_repeat": {
        "backbone": "efficientnet_b0",
        "pretrained": True,
        "image_size": 160,
        "batch_size": 16,
        "epochs": 8,
        "freeze_features_epochs": 2,
        "label_smoothing": 0.1,
        "random_erasing_prob": 0.0,
        "full_augmentation": True,
        "balanced_sampler": False,
        "class_weighted_loss": False,
        "best_metric": "accuracy",
    },
}


def build_minimal_transforms(image_size: int):
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
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


def create_custom_datasets(data_dir: Path, image_size: int, seed: int):
    from sklearn.model_selection import train_test_split

    base_dataset = ImageFolder(data_dir)
    train_transform, eval_transform = build_minimal_transforms(image_size)

    indices = list(range(len(base_dataset.samples)))
    labels = [label for _, label in base_dataset.samples]

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=0.1,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [labels[index] for index in train_val_indices]
    adjusted_val_ratio = 0.2 / 0.9
    train_indices, val_indices = train_test_split(
        train_val_indices,
        test_size=adjusted_val_ratio,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_dataset = TransformSubset(base_dataset, train_indices, train_transform)
    val_dataset = TransformSubset(base_dataset, val_indices, eval_transform)
    test_dataset = TransformSubset(base_dataset, test_indices, eval_transform)

    split_summary = {
        "class_to_idx": base_dataset.class_to_idx,
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "test_size": len(test_dataset),
        "total_size": len(base_dataset),
        "train_class_counts": {
            base_dataset.classes[class_index]: sum(1 for index in train_indices if labels[index] == class_index)
            for class_index in range(len(base_dataset.classes))
        },
    }
    return train_dataset, val_dataset, test_dataset, split_summary


def freeze_feature_extractor(model: nn.Module, backbone_name: str, trainable: bool) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = trainable

    if trainable:
        return

    if backbone_name in {"efficientnet_b0", "mobilenet_v3_small"}:
        for parameter in model.classifier.parameters():
            parameter.requires_grad = True
        return

    if backbone_name == "resnet18":
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return

    if backbone_name == "vit_tiny":
        head = getattr(model, "head", None)
        if head is not None:
            for parameter in head.parameters():
                parameter.requires_grad = True
            return
        heads = getattr(model, "heads", None)
        if heads is not None:
            for parameter in heads.parameters():
                parameter.requires_grad = True
            return
        raise AttributeError("Unable to locate ViT classifier head for freezing logic.")

    raise ValueError(f"Unknown backbone for freeze logic: {backbone_name}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a comparative flower experiment preset.")
    parser.add_argument("--preset", required=True, choices=sorted(PRESETS))
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = copy.deepcopy(PRESETS[args.preset])
    set_seed(args.seed)
    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    if config["full_augmentation"]:
        train_dataset, val_dataset, test_dataset, split_summary = create_datasets(
            data_dir=args.data_dir,
            image_size=config["image_size"],
            val_ratio=0.2,
            test_ratio=0.1,
            seed=args.seed,
            random_erasing_prob=config["random_erasing_prob"],
        )
    else:
        train_dataset, val_dataset, test_dataset, split_summary = create_custom_datasets(
            data_dir=args.data_dir,
            image_size=config["image_size"],
            seed=args.seed,
        )

    train_loader, val_loader, test_loader, class_weights = create_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=config["batch_size"],
        num_workers=args.num_workers,
        balanced_sampler=config["balanced_sampler"],
    )

    class_names = train_dataset.dataset.classes
    model = create_backbone(
        config["backbone"],
        num_classes=len(class_names),
        pretrained=config["pretrained"],
    ).to(device)

    if config["freeze_features_epochs"] > 0:
        freeze_feature_extractor(model, config["backbone"], trainable=False)

    loss_weights = None
    if config["class_weighted_loss"] and class_weights is not None:
        loss_weights = class_weights.to(device)

    criterion = nn.CrossEntropyLoss(
        weight=loss_weights,
        label_smoothing=config["label_smoothing"],
    )
    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=1e-4,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config["epochs"])

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": [],
        "val_macro_f1": [],
        "lr": [],
        "epoch_seconds": [],
    }
    best_state = None
    best_metric_value = -1.0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0

    for epoch in range(1, config["epochs"] + 1):
        if epoch == config["freeze_features_epochs"] + 1 and config["freeze_features_epochs"] > 0:
            freeze_feature_extractor(model, config["backbone"], trainable=True)
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=1e-4,
                weight_decay=1e-4,
            )
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(config["epochs"] - config["freeze_features_epochs"], 1),
            )

        start = time.perf_counter()
        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        epoch_seconds = time.perf_counter() - start
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])
        history["epoch_seconds"].append(epoch_seconds)

        print(
            f"Epoch {epoch:02d}/{config['epochs']} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"time={epoch_seconds:.2f}s | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        current_metric = val_macro_f1 if config["best_metric"] == "macro_f1" else val_acc
        if current_metric >= best_metric_value:
            best_metric_value = current_metric
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())

        scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_macro_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test | loss={test_loss:.4f} acc={test_acc:.4f} macro_f1={test_macro_f1:.4f}")

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": config["image_size"],
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "backbone": config["backbone"],
            "preset": args.preset,
            "config": config,
        },
        args.output_dir / "best_model.pt",
    )

    split_summary["config"] = config
    save_json(split_summary, args.output_dir / "split_summary.json")
    save_json(history, args.output_dir / "history.json")
    save_json(
        {
            "preset": args.preset,
            "backbone": config["backbone"],
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "best_metric": config["best_metric"],
            "avg_epoch_seconds": sum(history["epoch_seconds"]) / len(history["epoch_seconds"]),
            "epochs": config["epochs"],
            "image_size": config["image_size"],
            "batch_size": config["batch_size"],
            "pretrained": config["pretrained"],
            "freeze_features_epochs": config["freeze_features_epochs"],
            "label_smoothing": config["label_smoothing"],
            "random_erasing_prob": config["random_erasing_prob"],
            "full_augmentation": config["full_augmentation"],
            "balanced_sampler": config["balanced_sampler"],
            "class_weighted_loss": config["class_weighted_loss"],
        },
        args.output_dir / "metrics_summary.json",
    )
    save_training_plot(history, args.output_dir / "training_curve.png")

    cm = confusion_matrix(y_true, y_pred)
    save_confusion_matrix(cm, class_names, args.output_dir / "confusion_matrix.png")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    (args.output_dir / "classification_report.txt").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()
