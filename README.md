# BFIU-Net

三维地震断层分割模型及训练、评估、预测代码。

Code for 3D seismic fault segmentation, including model implementations, training, evaluation, and prediction.

## 模型 / Models

| 模型目录 / 命令行名称<br>Model directory / command-line name | 用途 | Purpose |
|---|---|---|
| `BFIU_Net` | 主模型 | Main model |
| `NIU-Net` | 无特征交互消融 | Ablation variant without cross-branch feature exchange |
| `LHU-Net` | 低分辨率→高分辨率交互消融 | Ablation variant with low-to-high-resolution feature exchange only |
| `HLU-Net` | 高分辨率→低分辨率交互消融 | Ablation variant with high-to-low-resolution feature exchange only |
| `U_Net` | U-Net 基线 | U-Net baseline |
| `W_NET` | W-Net 对比 | W-Net comparison model |
| `MS_UNET` | MS-UNet 对比 | MS-UNet comparison model |

BFIU-Net 使用三尺度编码—解码结构：32 通道浅层特征、64/128 通道双分支、A1/A2 残差块及双向特征交互。
低分辨率分支经 128→128 转置卷积对齐后，拼接高分辨率分支形成 192 通道；
再拼接编码特征形成 256 通道，经转置卷积解码及浅层跳跃连接，输出单通道断层概率。
默认输入为 `128 × 128 × 128 × 1`。

BFIU-Net uses a three-scale encoder–decoder architecture with 32-channel shallow features, parallel high- and low-resolution branches with 64 and 128 channels, A1/A2 residual blocks, and bidirectional feature interaction.
The low-resolution features are spatially aligned using a transposed convolution with 128 input and 128 output channels, then concatenated with the high-resolution features to form a 192-channel representation.
This representation is concatenated with the encoder features to form 256 channels. Decoding through transposed convolution and a shallow skip connection produces a single-channel fault probability volume.
The default input shape is `128 × 128 × 128 × 1`.

## 文件 / Files

```text
scripts/
  train.py               # 训练 / Training
  evaluate.py            # 评估 / Evaluation
  predict.py             # 预测 / Prediction
src/BFIU-Net/
  models/                # 模型 / Models
  arguments.py           # 命令行参数检查 / Command-line argument validation
  data.py                # 数据读取与批处理 / Data loading and batching
  losses.py              # 损失函数 / Loss functions
  metrics.py             # 指标 / Metrics
  training.py            # 训练 / Training
  evaluation.py          # 评估 / Evaluation
requirements.txt
README.md
```

## 安装 / Installation

使用 **Python 3.9 或 3.10**。在本文件所在目录执行：

Use **Python 3.9 or 3.10**. Run the following command from the directory containing this README:

```bash
python -m pip install -r requirements.txt
```
依赖版本为 TensorFlow 2.10、Keras 2.10、NumPy 1.23.5 和 h5py 3.12.1。

The dependencies are TensorFlow 2.10, Keras 2.10, NumPy 1.23.5, and h5py 3.12.1.

## 训练 / Training

```bash
python scripts/train.py --model BFIU_NET --output-dir outputs/BFIU_NET --epochs 40 --batch-size 1 --learning-rate 0.0001
```

常用可选参数：`--patch-size 128 128 128`、`--loss bce`、`--seed 42`、`--threshold 0.5`。
上述值是可调整的运行默认值。

Common optional arguments include `--patch-size 128 128 128`, `--loss bce`, `--seed 42`, and `--threshold 0.5`.
These are default runtime values and can be adjusted.

输出包括最佳权重 `best.weights.h5`、训练/验证指标 `history.csv`等。

Outputs include the best model weights (`best.weights.h5`) and training/validation metrics (`history.csv`).

## 评估 / Evaluation

```bash
python scripts/evaluate.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --output outputs/BFIU_NET/test_metrics.json
```

## 预测 / Prediction

```bash
python scripts/predict.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --input data/volume.npy --output outputs/prediction/probability.npy
```

## 相关参考 / References and Acknowledgments

工作流程参考 [FaultSeg3D](https://github.com/xinwucwp/faultSeg)，感谢FaultSeg3D的代码开源。
W-Net、MS-UNet 为项目中的论文适配实现，参考 [W-Net](https://doi.org/10.1109/LGRS.2024.3404505) 和 [MS-Unet](https://doi.org/10.3390/pr13071976)。

The workflow draws on [FaultSeg3D](https://github.com/xinwucwp/faultSeg). We thank its developers for making the code publicly available.
The W-Net and MS-UNet implementations in this repository are adapted from the methods described in the [W-Net](https://doi.org/10.1109/LGRS.2024.3404505) and [MS-Unet](https://doi.org/10.3390/pr13071976) papers.

真实地震数据体公开数据入口：[SEG Open Data](https://wiki.seg.org/wiki/Open_data)。

Public field seismic datasets can be accessed through [SEG Open Data](https://wiki.seg.org/wiki/Open_data).
