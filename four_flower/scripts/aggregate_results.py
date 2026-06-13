"""Aggregate experiment metrics into CSV tables and summary plots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

BACKBONE_PRESETS = [
    "eff_b0_full",
    "resnet18_full",
    "mobilenet_v3_small_full",
    "vit_tiny_full",
]
BACKBONE_DISPLAY = {
    "eff_b0_full": "EfficientNet-B0",
    "resnet18_full": "ResNet-18",
    "mobilenet_v3_small_full": "MobileNetV3-Small",
    "vit_tiny_full": "ViT-Tiny",
}

ABLATION_PRESETS = [
    "eff_b0_full",
    "ablate_no_pretrain",
    "ablate_no_freeze",
    "ablate_no_ls",
    "ablate_min_aug",
]
ABLATION_DISPLAY = {
    "eff_b0_full": "Full (anchor)",
    "ablate_no_pretrain": "- Pretrain",
    "ablate_no_freeze": "- Freeze stage",
    "ablate_no_ls": "- Label smoothing",
    "ablate_min_aug": "- Augmentation",
}

HYPERPARAM_PRESETS = [
    "hyper_h1_img128",
    "eff_b0_full",
    "hyper_h2_img192",
    "hyper_h3_bs8",
    "hyper_h4_bs32",
    "hyper_h5_repeat",
]
HYPERPARAM_DISPLAY = {
    "hyper_h1_img128": "img=128, bs=16",
    "eff_b0_full": "img=160, bs=16 (anchor)",
    "hyper_h2_img192": "img=192, bs=16",
    "hyper_h3_bs8": "img=160, bs=8",
    "hyper_h4_bs32": "img=160, bs=32",
    "hyper_h5_repeat": "img=160, bs=16 (repeat)",
}


def _load_metrics(run_dir: Path) -> dict | None:
    metrics_path = run_dir / "metrics_summary.json"
    if not metrics_path.exists():
        return None
    return json.loads(metrics_path.read_text(encoding="utf-8"))


def _write_csv(rows: list[dict], fieldnames: list[str], csv_out: Path) -> None:
    csv_out.parent.mkdir(parents=True, exist_ok=True)
    with csv_out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def aggregate_backbone_comparison(runs_dir: Path, csv_out: Path, png_out: Path) -> list[dict]:
    rows = []
    for preset in BACKBONE_PRESETS:
        metrics = _load_metrics(runs_dir / preset)
        if metrics is None:
            continue
        rows.append(
            {
                "backbone": BACKBONE_DISPLAY.get(preset, preset),
                "preset": preset,
                "test_acc": round(metrics["test_acc"], 4),
                "test_macro_f1": round(metrics["test_macro_f1"], 4),
            }
        )

    _write_csv(rows, ["backbone", "preset", "test_acc", "test_macro_f1"], csv_out)

    if rows:
        names = [row["backbone"] for row in rows]
        accs = [row["test_acc"] for row in rows]
        f1s = [row["test_macro_f1"] for row in rows]
        x = range(len(names))
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar([idx - 0.2 for idx in x], accs, width=0.4, label="Accuracy", color="C0")
        ax.bar([idx + 0.2 for idx in x], f1s, width=0.4, label="Macro F1", color="C1")
        ax.set_xticks(list(x))
        ax.set_xticklabels(names, rotation=20, ha="right")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Backbone Comparison (Test Set)")
        ax.legend()
        fig.tight_layout()
        png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return rows


def aggregate_ablation(runs_dir: Path, csv_out: Path, png_out: Path) -> list[dict]:
    rows = []
    for preset in ABLATION_PRESETS:
        metrics = _load_metrics(runs_dir / preset)
        if metrics is None:
            continue
        rows.append(
            {
                "config": ABLATION_DISPLAY.get(preset, preset),
                "preset": preset,
                "test_acc": round(metrics["test_acc"], 4),
                "test_macro_f1": round(metrics["test_macro_f1"], 4),
            }
        )

    _write_csv(rows, ["config", "preset", "test_acc", "test_macro_f1"], csv_out)

    if rows:
        names = [row["config"] for row in rows]
        accs = [row["test_acc"] for row in rows]
        f1s = [row["test_macro_f1"] for row in rows]
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(names, accs, color="C0", alpha=0.85, label="Accuracy")
        ax.plot(names, f1s, "o-", color="C1", label="Macro F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Ablation Study")
        plt.xticks(rotation=15, ha="right")
        ax.legend()
        fig.tight_layout()
        png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return rows


def aggregate_hyperparam(runs_dir: Path, csv_out: Path, png_out: Path) -> list[dict]:
    rows = []
    for preset in HYPERPARAM_PRESETS:
        metrics = _load_metrics(runs_dir / preset)
        if metrics is None:
            continue
        rows.append(
            {
                "config": HYPERPARAM_DISPLAY.get(preset, preset),
                "preset": preset,
                "test_acc": round(metrics["test_acc"], 4),
                "test_macro_f1": round(metrics["test_macro_f1"], 4),
            }
        )

    _write_csv(rows, ["config", "preset", "test_acc", "test_macro_f1"], csv_out)

    if rows:
        anchor_scores = [
            row["test_acc"]
            for row in rows
            if row["preset"] in {"eff_b0_full", "hyper_h5_repeat"}
        ]
        if len(anchor_scores) >= 2:
            mean_anchor = statistics.mean(anchor_scores)
            std_anchor = statistics.stdev(anchor_scores)
            print(
                "Anchor reproducibility | "
                f"mean_acc={mean_anchor:.4f} std_acc={std_anchor:.4f} n={len(anchor_scores)}"
            )

        names = [row["config"] for row in rows]
        accs = [row["test_acc"] for row in rows]
        f1s = [row["test_macro_f1"] for row in rows]
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(names, accs, "o-", color="C0", label="Accuracy")
        ax.plot(names, f1s, "s--", color="C1", label="Macro F1")
        ax.set_ylim(0.0, 1.0)
        ax.set_ylabel("Score")
        ax.set_title("Hyperparameter Sensitivity")
        plt.xticks(rotation=20, ha="right")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        png_out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(png_out, dpi=200, bbox_inches="tight")
        plt.close(fig)

    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate flower experiment results.")
    parser.add_argument("--runs-dir", type=Path, default=Path("four_flower/runs"))
    parser.add_argument("--task", choices=["backbone", "ablation", "hyperparam"], required=True)
    parser.add_argument("--csv-out", type=Path, required=True)
    parser.add_argument("--png-out", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.task == "backbone":
        rows = aggregate_backbone_comparison(args.runs_dir, args.csv_out, args.png_out)
    elif args.task == "ablation":
        rows = aggregate_ablation(args.runs_dir, args.csv_out, args.png_out)
    else:
        rows = aggregate_hyperparam(args.runs_dir, args.csv_out, args.png_out)
    print(f"Wrote {args.csv_out} with {len(rows)} rows and {args.png_out}")


if __name__ == "__main__":
    main()
