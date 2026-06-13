# EfficientNet-B0 Final Experiment Results

## Experiment environment

- Runtime environment: `conda` environment `yolov11_test`
- Python version: `3.9.23`
- Device: `CPU`
- Dataset size: `3037`
- Train/validation/test split: `2125 / 608 / 304`
- Classes: `dandelion`, `roses`, `sunflowers`, `tulips`

## Experiment settings

### Baseline experiment

- Model: `EfficientNet-B0`
- Weight initialization: random initialization
- Image size: `160`
- Batch size: `16`
- Epochs: `8`
- Label smoothing: `0.1`
- Learning-rate scheduler: `cosine`

Command:

```bash
D:\Anaconda\envs\yolov11_test\python.exe train_efficientnet.py --data-dir input_data --output-dir runs/efficientnet_baseline_8ep --epochs 8 --batch-size 16 --image-size 160 --weights none --label-smoothing 0.1 --num-workers 0 --scheduler cosine
```

### Final experiment

- Model: `EfficientNet-B0`
- Weight initialization: ImageNet pretrained weights
- Training strategy: freeze feature extractor for the first `2` epochs, then unfreeze for fine-tuning
- Image size: `160`
- Batch size: `16`
- Epochs: `8`
- Label smoothing: `0.1`
- Learning-rate scheduler: `cosine`

Command:

```bash
D:\Anaconda\envs\yolov11_test\python.exe train_efficientnet.py --data-dir input_data --output-dir runs/efficientnet_final --epochs 8 --batch-size 16 --image-size 160 --weights default --freeze-features-epochs 2 --label-smoothing 0.1 --num-workers 0 --scheduler cosine
```

## Core results

| Experiment | Validation accuracy | Test accuracy | Macro F1 |
| --- | ---: | ---: | ---: |
| Baseline | 0.6990 | 0.6842 | 0.6647 |
| Final | 0.8684 | 0.8487 | 0.8432 |
| Optimized Final | 0.8586 | 0.8750 | 0.8703 |

- Absolute improvement in test accuracy: `0.1645`
- Relative improvement in test accuracy: `24.04%`
- Optimized-vs-final absolute improvement in test accuracy: `0.0263`
- Optimized-vs-final absolute improvement in macro F1: `0.0271`

## Optimized experiment

- Model: `EfficientNet-B0`
- Weight initialization: ImageNet pretrained weights
- Training strategy: freeze feature extractor for the first `2` epochs, then unfreeze for fine-tuning
- Image size: `160`
- Batch size: `16`
- Epochs: `10`
- Label smoothing: `0.1`
- Balanced sampler: enabled
- Class-weighted loss: enabled
- Random erasing: `0.15`
- Best checkpoint metric: `macro_f1`
- Learning-rate scheduler: `cosine`

Command:

```bash
D:\Anaconda\envs\yolov11_test\python.exe train_efficientnet.py --data-dir input_data --output-dir runs/efficientnet_tuned_v1 --epochs 10 --batch-size 16 --image-size 160 --weights default --freeze-features-epochs 2 --label-smoothing 0.1 --random-erasing-prob 0.15 --balanced-sampler --class-weighted-loss --best-metric macro_f1 --num-workers 0 --scheduler cosine
```

## Per-class metrics of the final experiment

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| dandelion | 0.8925 | 0.9222 | 0.9071 |
| roses | 0.7742 | 0.7500 | 0.7619 |
| sunflowers | 0.8873 | 0.9000 | 0.8936 |
| tulips | 0.8205 | 0.8000 | 0.8101 |

## Per-class metrics of the optimized final experiment

| Class | Precision | Recall | F1-score |
| --- | ---: | ---: | ---: |
| dandelion | 0.9053 | 0.9556 | 0.9297 |
| roses | 0.8525 | 0.8125 | 0.8320 |
| sunflowers | 0.8553 | 0.9286 | 0.8904 |
| tulips | 0.8750 | 0.7875 | 0.8289 |

## Output artifacts

Baseline experiment outputs:

- `runs/efficientnet_baseline_8ep/best_model.pt`
- `runs/efficientnet_baseline_8ep/training_curve.png`
- `runs/efficientnet_baseline_8ep/confusion_matrix.png`
- `runs/efficientnet_baseline_8ep/classification_report.txt`

Final experiment outputs:

- `runs/efficientnet_final/best_model.pt`
- `runs/efficientnet_final/training_curve.png`
- `runs/efficientnet_final/confusion_matrix.png`
- `runs/efficientnet_final/classification_report.txt`

Optimized final experiment outputs:

- `runs/efficientnet_tuned_v1/best_model.pt`
- `runs/efficientnet_tuned_v1/training_curve.png`
- `runs/efficientnet_tuned_v1/confusion_matrix.png`
- `runs/efficientnet_tuned_v1/classification_report.txt`
- `runs/efficientnet_tuned_v1/metrics_summary.json`

## Conclusion

Under the current local CPU environment, the pretrained EfficientNet-B0 with staged fine-tuning significantly outperformed the randomly initialized baseline. After further enabling balanced sampling, class-weighted loss, random erasing, and macro-F1-based checkpoint selection, the optimized final experiment improved test accuracy from `84.87%` to `87.50%`, showing that the main gains came from better handling of class imbalance and better model-selection criteria rather than simply increasing model size.
