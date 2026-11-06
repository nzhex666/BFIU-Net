# BFIU-Net

三维地震断层分割模型及训练、评估、预测代码。

## 模型

| 模型目录 / 命令行名称 | 用途 |
|---|---|
| `BFIU_Net` | 主模型 |
| `NIU-Net` | 无特征交互消融 |
| `LHU-Net` | 低分辨率→高分辨率交互消融 |
| `HLU-Net` | 高分辨率→低分辨率交互消融 |
| `U_Net` | U-Net 基线 |
| `W_NET` | W-Net 对比 |
| `MS_UNET` | MS-UNet 对比 |

BFIU-Net 使用三尺度编码—解码结构：32 通道浅层特征、64/128 通道双分支、A1/A2 残差块及双向特征交互。
低分辨率分支经 128→128 转置卷积对齐后，拼接高分辨率分支形成 192 通道；
再拼接编码特征形成 256 通道，经转置卷积解码及浅层跳跃连接，输出单通道断层概率。
默认输入为 `128 × 128 × 128 × 1`。

## 文件

```text
scripts/
  train.py               # 训练
  evaluate.py            # 评估
  predict.py             # 预测
src/BFIU-Net/
  models/                # 模型
  arguments.py           # 命令行参数检查
  data.py                # 数据读取与批处理
  losses.py              # 损失函数
  metrics.py             # 指标
  training.py            # 训练
  evaluation.py          # 评估
requirements.txt
README.md
```

## 安装

使用 **Python 3.9 或 3.10**。在本文件所在目录执行：
```bash
python -m pip install -r requirements.txt
```
依赖版本为 TensorFlow 2.10、Keras 2.10、NumPy 1.23.5 和 h5py 3.12.1。

## 训练

```bash
python scripts/train.py --model BFIU_NET --output-dir outputs/BFIU_NET --epochs 40 --batch-size 1 --learning-rate 0.0001
```

常用可选参数：`--patch-size 128 128 128`、`--loss bce`、`--seed 42`、`--threshold 0.5`。
上述值是可调整的运行默认值。

输出包括最佳权重 `best.weights.h5`、训练/验证指标 `history.csv`等。

## 评估

```bash
python scripts/evaluate.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --output outputs/BFIU_NET/test_metrics.json
```

## 预测

```bash
python scripts/predict.py --model BFIU_NET --weights outputs/BFIU_NET/best.weights.h5 --input data/volume.npy --output outputs/prediction/probability.npy
```

## 相关参考

工作流程参考 [FaultSeg3D](https://github.com/xinwucwp/faultSeg)，感谢FaultSeg3D的代码开源。
W-Net、MS-UNet 为项目中的论文适配实现，参考 [W-Net](https://doi.org/10.1109/LGRS.2024.3404505) 和 [MS-Unet](https://doi.org/10.3390/pr13071976)。

真实地震数据体公开数据入口：[SEG Open Data](https://wiki.seg.org/wiki/Open_data)。
