# BFIU-Net

**English** | [简体中文](README_zh-CN.md)

Code for 3D seismic fault segmentation, including model implementations, training, evaluation, and prediction.

## Models

| Model directory / command-line name | Purpose |
|---|---|
| `BFIU_Net` | Main model |
| `NIU-Net` | Ablation variant without cross-branch feature exchange |
| `LHU-Net` | Ablation variant with low-to-high-resolution feature exchange only |
| `HLU-Net` | Ablation variant with high-to-low-resolution feature exchange only |
| `U_Net` | U-Net baseline |
| `W_NET` | W-Net comparison model |
| `MS_UNET` | MS-UNet comparison model |

BFIU-Net uses a three-scale encoder–decoder architecture with 32-channel shallow features, parallel high- and low-resolution branches with 64 and 128 channels, A1/A2 residual blocks, and bidirectional feature interaction.
The low-resolution features are spatially aligned using a transposed convolution with 128 input and 128 output channels, then concatenated with the high-resolution features to form a 192-channel representation.
This representation is concatenated with the encoder features to form 256 channels. Decoding through transposed convolution and a shallow skip connection produces a single-channel fault probability volume.
The default input shape is `128 × 128 × 128 × 1`.

## Files

```text
scripts/
  train.py               # Training
  evaluate.py            # Evaluation
  predict.py             # Prediction
src/BFIU-Net/
  models/                # Models
  arguments.py           # Command-line argument validation
  data.py                # Data loading and batching
  losses.py              # Loss functions
  metrics.py             # Metrics
  training.py            # Training
  evaluation.py          # Evaluation
requirements.txt
README.md
README_zh-CN.md
```

## Installation

Use **Python 3.9 or 3.10**. Run the following command from the directory containing this README:

```bash
python -m pip install -r requirements.txt
```

The dependencies are TensorFlow 2.10, Keras 2.10, NumPy 1.23.5, and h5py 3.12.1.

## Training

```bash
python scripts/train.py --model BFIU_NET --output-dir outputs/BFIU_NET --epochs 40 --batch-size 1 --learning-rate 0.0001
```

Common optional arguments include `--patch-size 128 128 128`, `--loss bce`, `--seed 42`, and `--threshold 0.5`.
These are default runtime values and can be adjusted.

Outputs include the best model weights (`best.weights.h5`) and training/validation metrics (`history.csv`).

## Evaluation

```bash
python scripts/evaluate.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --output outputs/BFIU_NET/test_metrics.json
```

## Prediction

```bash
python scripts/predict.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --input data/volume.npy --output outputs/prediction/probability.npy
```

## References and Acknowledgments

The workflow draws on [FaultSeg3D](https://github.com/xinwucwp/faultSeg). We thank its developers for making the code publicly available.
The W-Net and MS-UNet implementations in this repository are adapted from the methods described in the [W-Net](https://doi.org/10.1109/LGRS.2024.3404505) and [MS-Unet](https://doi.org/10.3390/pr13071976) papers.

Public field seismic datasets can be accessed through [SEG Open Data](https://wiki.seg.org/wiki/Open_data).
