from __future__ import annotations

import argparse
from pathlib import Path

import torch

try:
    from four_flower.efficientnet_utils import create_model, get_device, load_image
except ModuleNotFoundError:
    from efficientnet_utils import create_model, get_device, load_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict a flower category with EfficientNet-B0.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to best_model.pt")
    parser.add_argument("--image", type=Path, required=True, help="Path to the input image")
    parser.add_argument("--top-k", type=int, default=3)
    return parser.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    class_names = checkpoint["class_names"]
    image_size = checkpoint.get("image_size", 224)

    model = create_model(
        num_classes=len(class_names),
        use_pretrained=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    device = get_device()
    model.to(device)
    model.eval()

    image_tensor = load_image(args.image, image_size=image_size).to(device)
    logits = model(image_tensor)
    probabilities = torch.softmax(logits, dim=1)[0]
    top_probabilities, top_indices = torch.topk(probabilities, k=min(args.top_k, len(class_names)))

    print(f"Image: {args.image}")
    print("Prediction results:")
    for probability, index in zip(top_probabilities.cpu().tolist(), top_indices.cpu().tolist()):
        print(f"  {class_names[index]}: {probability:.4f}")


if __name__ == "__main__":
    main()
