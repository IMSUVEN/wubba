<h1 align="center">🧪 Wubba</h1>

<p align="center">
  <strong>Web Understanding By Behavioral Augmentation</strong><br>
  <sub>HTML 文档的自监督表示学习</sub>
</p>

<p align="center">
  <a href="./README.md">English</a> | 简体中文
</p>

<p align="center">
  <a href="#-安装">安装</a> •
  <a href="#-快速开始">快速开始</a> •
  <a href="#-特性">特性</a> •
  <a href="#-架构">架构</a> •
  <a href="#-许可证">许可证</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.10+-3776ab?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.7+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Lightning-2.5+-792ee5?logo=lightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/badge/code%20style-ruff-000000?logo=ruff&logoColor=white" alt="Ruff">
  <img src="https://img.shields.io/badge/types-pyright-blue?logo=python&logoColor=white" alt="Pyright">
  <img src="https://img.shields.io/badge/license-MIT-green.svg" alt="License">
</p>

---

Wubba 使用对比学习从原始 HTML 中学习**布局不变的嵌入表示**。可将任意 HTML 文档转换为固定大小的向量，用于相似度搜索、聚类或分类。

```python
from wubba import WubbaInference

model = WubbaInference("model.ckpt")
embeddings = model.predict(["<html>...</html>", "<html>...</html>"])
similarity = model.compute_similarity(html1, html2)
```

## 📦 安装

```bash
uv sync                # 或: pip install .
uv sync --group dev    # 包含开发工具
uv sync --extra onnx   # 包含 ONNX 导出支持
```

## 🚀 快速开始

### 🔍 生成嵌入向量

```python
from wubba import WubbaInference

model = WubbaInference("models/best.ckpt")

# 完整嵌入
embeddings = model.predict(html_docs)  # (N, 256)

# Matryoshka：截断以权衡速度/大小
embeddings = model.predict(html_docs, dim=64)  # (N, 64)
```

### 🎯 训练模型

```python
from wubba import Config, train

config = Config(
    data_dir="data/",
    num_epochs=100,
    batch_size=1024,
    loss_type="enhanced_hybrid",
)

model, trainer = train(config)
```

### 🚢 部署

```python
model = WubbaInference("model.ckpt")
model.quantize()                    # INT8 量化加速 CPU 推理
model.export_onnx("model.onnx")     # 跨平台部署
```

## ✨ 特性

| 类别 | 特性 |
|------|------|
| ⚡ **性能** | Flash Attention (SDPA)、`torch.compile()`、INT8 量化、ONNX 导出 |
| 🧠 **架构** | 层次化 RoPE、多头注意力 (SDPA)、RMSNorm + SwiGLU |
| 🎯 **嵌入** | Matryoshka（32/64/128/256 维）、布局不变表示 |
| 📉 **损失函数** | VICReg、InfoNCE、谱对比损失、困难负样本挖掘、对齐-均匀性 |
| 🎓 **训练** | 课程学习、自步学习、EMA、表示坍塌检测 |
| 🔧 **多任务** | 掩码节点预测、结构预测（深度与计数） |
| 🌳 **数据增强** | 上下文感知增强、Tree Mixup、语义替换、子树打乱 |

## 🏗️ 架构

```
HTML → 解析器 → 节点特征 → Transformer 编码器 → CLS 池化 → 嵌入向量
                   ↑                    ↑
            10/15 维特征          层次化 RoPE
                                  Flash Attention
                                  RMSNorm + SwiGLU
```

📊 **每个节点的输入特征：** tag_id、语义分组、深度、位置、子节点数、兄弟节点数、是否叶节点、父节点 tag_id、标签角色、子树深度

## ⚙️ 配置

`Config` 中的关键选项：

```python
Config(
    # 🧠 模型
    transformer_dim=256,
    transformer_layers=6,
    matryoshka_dims=[32, 64, 128, 256],
    
    # 🎯 训练
    loss_type="enhanced_hybrid",  # vicreg | infonce | hybrid | matryoshka_hybrid | enhanced_hybrid
    use_ema=True,
    enable_multitask=True,
    
    # 📊 数据
    use_extended_features=True,   # 15 维特征
    use_contextual_aug=True,
)
```

## 📚 示例

| 示例 | 描述 |
|------|------|
| [01_quickstart.py](examples/01_quickstart.py) | 快速训练并生成嵌入向量 |
| [02_web_deduplication.py](examples/02_web_deduplication.py) | 为爬虫检测重复/相似页面 |
| [03_similarity_search.py](examples/03_similarity_search.py) | 构建 HTML 文档搜索索引 |
| [04_page_classification.py](examples/04_page_classification.py) | 使用嵌入作为特征进行页面分类 |
| [05_production_deployment.py](examples/05_production_deployment.py) | 量化与 ONNX 导出 |
| [06_embedding_analysis.py](examples/06_embedding_analysis.py) | 分析嵌入质量并可视化 |
| [07_custom_training.py](examples/07_custom_training.py) | 使用自定义回调的高级训练 |
| [08_batch_processing.py](examples/08_batch_processing.py) | 批量处理百万级 HTML 文档 |

## 📁 项目结构

```
src/wubba/
├── config.py      # 所有超参数
├── model.py       # 编码器和损失函数
├── data.py        # 数据处理和增强
├── train.py       # 训练流程
├── inference.py   # 推理和导出
├── metrics.py     # 嵌入质量指标
└── utils.py       # DOM 工具函数
```

## 🛠️ 开发

```bash
uv run ruff format src    # 🎨 格式化
uv run ruff check src     # 🔍 代码检查
uv run pyright            # 📝 类型检查
```

📖 详细开发指南请参阅 [AGENTS.md](AGENTS.md)。

## 📄 许可证

MIT
