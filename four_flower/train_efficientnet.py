from __future__ import annotations

import argparse
import copy
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.datasets import ImageFolder
from torchvision.datasets.folder import default_loader

try:
    from four_flower.efficientnet_utils import create_model, get_device, save_json, set_seed
except ModuleNotFoundError:
    from efficientnet_utils import create_model, get_device, save_json, set_seed


class TransformSubset(Dataset):
    def __init__(self, dataset: ImageFolder, indices: list[int], transform):
        self.dataset = dataset
        self.indices = indices
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int):
        image_path, label = self.dataset.samples[self.indices[idx]]
        image = default_loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label


def build_transforms(image_size: int = 224, random_erasing_prob: float = 0.0):
    try:
        from four_flower.efficientnet_utils import build_transforms as build_default_transforms
    except ModuleNotFoundError:
        from efficientnet_utils import build_transforms as build_default_transforms

    return build_default_transforms(image_size=image_size, random_erasing_prob=random_erasing_prob)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an EfficientNet-B0 flower classifier.")
    parser.add_argument("--data-dir", type=Path, required=True, help="Path to the extracted flower dataset.")
    parser.add_argument("--output-dir", type=Path, default=Path("runs/efficientnet_b0"))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--freeze-features-epochs", type=int, default=0)
    parser.add_argument("--label-smoothing", type=float, default=0.0)
    parser.add_argument("--random-erasing-prob", type=float, default=0.0)
    parser.add_argument("--balanced-sampler", action="store_true")
    parser.add_argument("--class-weighted-loss", action="store_true")
    parser.add_argument(
        "--best-metric",
        choices=["accuracy", "macro_f1"],
        default="accuracy",
        help="Metric used to keep the best checkpoint.",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine"],
        default="cosine",
        help="Learning-rate scheduler used during training.",
    )
    parser.add_argument(
        "--weights",
        choices=["default", "none"],
        default="default",
        help="Use ImageNet pretrained weights when available.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Dataset directory not found: {args.data_dir}")
    if args.val_ratio <= 0 or args.test_ratio <= 0:
        raise ValueError("Validation and test ratios must be greater than 0.")
    if args.val_ratio + args.test_ratio >= 1:
        raise ValueError("Validation ratio plus test ratio must be less than 1.")


def create_datasets(data_dir: Path, image_size: int, val_ratio: float, test_ratio: float, seed: int, random_erasing_prob: float):
    train_transform, eval_transform = build_transforms(
        image_size=image_size,
        random_erasing_prob=random_erasing_prob,
    )
    base_dataset = ImageFolder(data_dir)

    indices = list(range(len(base_dataset.samples)))
    labels = [label for _, label in base_dataset.samples]

    train_val_indices, test_indices = train_test_split(
        indices,
        test_size=test_ratio,
        random_state=seed,
        stratify=labels,
    )
    train_val_labels = [labels[index] for index in train_val_indices]
    adjusted_val_ratio = val_ratio / (1 - test_ratio)
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


def build_train_sampler(train_dataset: TransformSubset):
    train_labels = [train_dataset.dataset.samples[index][1] for index in train_dataset.indices]
    class_counts = torch.bincount(torch.tensor(train_labels), minlength=len(train_dataset.dataset.classes)).float()
    class_weights = torch.where(class_counts > 0, 1.0 / class_counts, torch.zeros_like(class_counts))
    sample_weights = class_weights[torch.tensor(train_labels)].double()
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    return sampler, class_weights


def create_loaders(train_dataset, val_dataset, test_dataset, batch_size: int, num_workers: int, balanced_sampler: bool):
    train_sampler = None
    class_weights = None
    shuffle = True
    if balanced_sampler:
        train_sampler, class_weights = build_train_sampler(train_dataset)
        shuffle = False

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=train_sampler,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    return train_loader, val_loader, test_loader, class_weights


def set_feature_extractor_trainable(model: nn.Module, trainable: bool) -> None:
    for param in model.features.parameters():
        param.requires_grad = trainable


def run_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    running_correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        preds = outputs.argmax(dim=1)
        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, running_correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_correct = 0
    total = 0
    y_true = []
    y_pred = []

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)
        loss = criterion(outputs, labels)
        preds = outputs.argmax(dim=1)

        running_loss += loss.item() * images.size(0)
        running_correct += (preds == labels).sum().item()
        total += labels.size(0)

        y_true.extend(labels.cpu().tolist())
        y_pred.extend(preds.cpu().tolist())

    macro_f1 = f1_score(y_true, y_pred, average="macro")
    return running_loss / total, running_correct / total, macro_f1, y_true, y_pred


def save_training_plot(history: dict, output_path: Path) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    plt.figure(figsize=(10, 4))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["train_loss"], label="train_loss")
    plt.plot(epochs, history["val_loss"], label="val_loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["train_acc"], label="train_acc")
    plt.plot(epochs, history["val_acc"], label="val_acc")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def save_confusion_matrix(cm, class_names: list[str], output_path: Path) -> None:
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.colorbar()
    plt.xticks(range(len(class_names)), class_names, rotation=30)
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")

    max_value = cm.max() if cm.size else 0
    threshold = max_value / 2 if max_value else 0
    for row in range(cm.shape[0]):
        for col in range(cm.shape[1]):
            plt.text(
                col,
                row,
                str(cm[row, col]),
                ha="center",
                va="center",
                color="white" if cm[row, col] > threshold else "black",
            )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def main() -> None:
    args = parse_args()
    validate_args(args)
    set_seed(args.seed)

    device = get_device()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset, val_dataset, test_dataset, split_summary = create_datasets(
        data_dir=args.data_dir,
        image_size=args.image_size,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        random_erasing_prob=args.random_erasing_prob,
    )
    train_loader, val_loader, test_loader, class_weights = create_loaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        balanced_sampler=args.balanced_sampler,
    )

    class_names = train_dataset.dataset.classes
    model = create_model(
        num_classes=len(class_names),
        use_pretrained=args.weights == "default",
    ).to(device)
    loss_weights = None
    if args.class_weighted_loss:
        if class_weights is None:
            _, class_weights = build_train_sampler(train_dataset)
        loss_weights = class_weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=args.label_smoothing)

    feature_trainable = args.freeze_features_epochs == 0
    set_feature_extractor_trainable(model, trainable=feature_trainable)

    optimizer = torch.optim.AdamW(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    scheduler = None
    if args.scheduler == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_state = None
    best_metric_value = -1.0
    best_val_acc = 0.0
    best_val_macro_f1 = 0.0
    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_macro_f1": [], "lr": []}

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_features_epochs + 1 and args.freeze_features_epochs > 0:
            set_feature_extractor_trainable(model, trainable=True)
            optimizer = torch.optim.AdamW(
                filter(lambda parameter: parameter.requires_grad, model.parameters()),
                lr=args.learning_rate * 0.1,
                weight_decay=args.weight_decay,
            )
            if args.scheduler == "cosine":
                remaining_epochs = max(args.epochs - args.freeze_features_epochs, 1)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=remaining_epochs)

        train_loss, train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_macro_f1, _, _ = evaluate(model, val_loader, criterion, device)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)
        history["val_macro_f1"].append(val_macro_f1)
        history["lr"].append(optimizer.param_groups[0]["lr"])

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"lr={optimizer.param_groups[0]['lr']:.6f} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_macro_f1={val_macro_f1:.4f}"
        )

        current_metric = val_macro_f1 if args.best_metric == "macro_f1" else val_acc
        if current_metric >= best_metric_value:
            best_metric_value = current_metric
            best_val_acc = val_acc
            best_val_macro_f1 = val_macro_f1
            best_state = copy.deepcopy(model.state_dict())

        if scheduler is not None:
            scheduler.step()

    if best_state is None:
        raise RuntimeError("Training did not produce a valid checkpoint.")

    model.load_state_dict(best_state)
    test_loss, test_acc, test_macro_f1, y_true, y_pred = evaluate(model, test_loader, criterion, device)
    print(f"Test loss={test_loss:.4f} test_acc={test_acc:.4f} test_macro_f1={test_macro_f1:.4f}")

    checkpoint_path = args.output_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "class_names": class_names,
            "image_size": args.image_size,
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "weights": args.weights,
            "best_metric": args.best_metric,
        },
        checkpoint_path,
    )

    save_json(split_summary, args.output_dir / "split_summary.json")
    save_json(history, args.output_dir / "history.json")
    save_json(
        {
            "best_val_acc": best_val_acc,
            "best_val_macro_f1": best_val_macro_f1,
            "test_loss": test_loss,
            "test_acc": test_acc,
            "test_macro_f1": test_macro_f1,
            "best_metric": args.best_metric,
            "balanced_sampler": args.balanced_sampler,
            "class_weighted_loss": args.class_weighted_loss,
            "random_erasing_prob": args.random_erasing_prob,
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
