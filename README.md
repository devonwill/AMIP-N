# AMIP-N

Attention-Mixed Prediction Network（AMIP-N）的代码实现。

## 项目简介

AMIP-N 是一个面向时间序列预测的深度学习模型，融合了**时序卷积网络（TCN）**、**小波多头自注意力机制（Wavelet Multi-Head Self-Attention）** 与 **LSTM**，以捕捉时序数据中的长短期依赖与多尺度频率特征。

模型的核心结构包括：

- **TCN 模块**：三层 `TemporalConvNet`，用于提取局部时序特征与扩大感受野。
- **注意力模块**：默认使用基于离散小波变换（DWT）的 `WaveletMultiHeadAttention`，并提供以下可替换的注意力机制：
  - 离散余弦多头自注意力（`DiscreteCosineMultiHeadSelfAttention`）
  - 傅里叶多头自注意力（`FourierMultiHeadSelfAttention`）
  - 多小波多头自注意力（`MultiWaveletMutiHeadSelfAttention`）
  - 傅里叶交叉注意力 / 自相关（`FourierCrossAttention` / `AutoCorrelationLayer`）
  - 稀疏概率注意力（`ProbAttention`、`FourierProbAttention` 等）
- **LSTM 模块**：三层堆叠的 LSTM，用于建模长程时序依赖。
- **全连接层**：将 LSTM 输出映射为最终的预测序列。

## 目录结构

```
AMIP-N/
├── amip_n.py              # AMIP-N 主模型定义
├── models/                # 模型副本（amip_n / gru / lstm / tcn）
├── attention/             # 多种注意力机制实现
│   ├── Wavelet_Multi_Head_Self_Attention.py
│   ├── WaveletMutiHeadSelfAttention.py
│   ├── FourierMutiHeadSelfAttention.py
│   ├── FourierCorrelation.py
│   ├── DCTMutiHeadSelfAttention.py
│   ├── MultiWaveletCorrelation.py
│   ├── AutoCorrelation.py
│   ├── MutiHeadSelfAttention.py
│   ├── SelfAttention_Family.py
│   └── attn.py
├── DWT/                   # 离散小波变换（1D / 2D 正逆变换）
│   ├── DWT.py
│   ├── DWT_layer.py
│   └── __init__.py
├── gru.py  lstm.py  tcn.py
└── README.md
```

## 环境依赖

- Python 3.7+
- PyTorch
- NumPy
- PyWavelets (`pywt`)
- SciPy
- SymPy
- einops
- Matplotlib

## 安装方法

### 1. 克隆仓库

```bash
git clone <repository-url>
cd AMIP-N
```

### 2. 创建并激活虚拟环境（推荐）

```bash
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows
```

### 3. 安装依赖

```bash
pip install torch numpy PyWavelets scipy sympy einops matplotlib
```

> 说明：模型中 `from models.TCN.tcn_my import TemporalConvNet` 与 `from utils.masking import ...` 引用了未包含在本仓库中的辅助模块。若要完整运行，请自行补充对应的 `TCN/tcn_my.py` 与 `utils/masking.py`，或根据实际路径调整 import。

## 快速使用

```python
import torch
from amip_n import AMIP_N

# length_in: 输入序列长度, feature_in: 特征维度, length_out: 预测长度
model = AMIP_N(
    length_in=96,
    feature_in=7,
    length_out=24,
    ker_size=3,
    num_channels=[64, 128, 256],
    num_hidden=[128, 128, 128],
    num_heads=8,
    dp=0.3,
).double()

x = torch.randn(32, 96, 7).double()   # (batch, seq_len, features)
y = model(x)                            # (batch, length_out, 1)
print(y.shape)
```

