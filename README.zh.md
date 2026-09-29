<p align="center">
  <a href="README.md">English</a> · <b>简体中文</b>
</p>

<div align="center">
<img src="https://raw.githubusercontent.com/google/flax/main/images/flax_logo_250px.png" alt="logo"></img>
</div>

# Flax：专为灵活性而设计的 JAX 神经网络库与生态系统

[![Flax - Test](https://github.com/google/flax/actions/workflows/flax_test.yml/badge.svg)](https://github.com/google/flax/actions/workflows/flax_test.yml)
[![PyPI version](https://img.shields.io/pypi/v/flax)](https://pypi.org/project/flax/)

[**概述**](#概述)
| [**快速安装**](#快速安装)
| [**Flax 代码范式**](#flax-代码范式)
| [**官方文档**](https://flax.readthedocs.io/)

Flax NNX 发布于 2024 年，是全新的简化版 Flax API，旨在让在 [JAX](https://jax.readthedocs.io/) 中创建、检查、调试和分析神经网络变得更加轻松自然。它通过为 Python 引用语义提供一等公民（First-class）支持来实现这一目标。这使得开发者可以使用标准 Python 对象来表达模型，从而直接实现引用共享与对象可变性（Mutability）。

Flax NNX 演进自 [Flax Linen API](https://flax-linen.readthedocs.io/)，后者由 Google Brain 的工程师与研究员与 JAX 团队密切合作于 2020 年发布。

您可以在 [Flax 官方文档站点](https://flax.readthedocs.io/) 深入了解 Flax NNX。推荐查阅：

* [Flax NNX 基础](https://flax.readthedocs.io/en/latest/nnx_basics.html)
* [MNIST 实战教程](https://flax.readthedocs.io/en/latest/mnist_tutorial.html)
* [为什么选择 Flax NNX](https://flax.readthedocs.io/en/latest/why.html)
* [从 Flax Linen 演进至 Flax NNX 指南](https://flax.readthedocs.io/en/latest/guides/linen_to_nnx.html)

**注：** Flax Linen 的[文档拥有独立的专属站点](https://flax-linen.readthedocs.io/)。

Flax 团队的使命是服务于不断壮大的 JAX 神经网络研究生态系统——涵盖 Alphabet 内部以及更广泛的全球开源社区，并探索 JAX 大放异彩的核心应用场景。我们几乎所有的协调、规划以及未来架构设计讨论都通过 GitHub 开展。我们真诚欢迎在 Discussion、Issue 和 Pull Request 中提供宝贵反馈与交流。

欢迎在我们的 [Flax GitHub Discussion 论坛](https://github.com/google/flax/discussions) 中提出功能需求、分享您正在开展的工作、报告问题或进行技术咨询。

我们将持续迭代完善 Flax，但预计不会对核心 API 做出重大的破坏性变更（Breaking changes）。我们会尽可能提供详细的[更新日志（Changelog）](https://github.com/google/flax/tree/main/CHANGELOG.md)以及废弃警告（Deprecation warnings）。

如需直接联系开发团队，可发送邮件至 flax-dev@google.com。

## 概述

Flax 是一个专为 JAX 打造的高性能神经网络库与生态系统，**专为灵活性而设计**：通过 Fork 示例并自由修改训练循环来尝试崭新的训练方式，而非受限于框架本身的特性束缚。

Flax 与 JAX 官方团队紧密合作联合开发，开箱即用，提供了开启研究所需的一切基础设施，包括：

* **神经网络 API** (`flax.nnx`): 包含 [`Linear`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Linear), [`Conv`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/linear.html#flax.nnx.Conv), [`BatchNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.BatchNorm), [`LayerNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.LayerNorm), [`GroupNorm`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/normalization.html#flax.nnx.GroupNorm), [注意力机制 Attention](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html) ([`MultiHeadAttention`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/attention.html#flax.nnx.MultiHeadAttention)), [`LSTMCell`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/recurrent.html#flax.nnx.nn.recurrent.LSTMCell), [`GRUCell`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/recurrent.html#flax.nnx.nn.recurrent.GRUCell), [`Dropout`](https://flax.readthedocs.io/en/latest/api_reference/flax.nnx/nn/stochastic.html#flax.nnx.Dropout)。

* **实用工具与工程范式**: 多设备复制训练（Replicated training）、序列化与检查点保存（Checkpointing）、评测指标（Metrics）、设备端预取（Prefetching on device）。

* **教学示例**: [MNIST 基础实战](https://flax.readthedocs.io/en/latest/mnist_tutorial.html)、[基于 Gemma 语言模型（Transformer）的推理与采样](https://github.com/google/flax/tree/main/examples/gemma)。

## 快速安装

Flax 基于 JAX 构建，请务必查看 [JAX 在 CPU、GPU 和 TPU 上的安装说明](https://jax.readthedocs.io/en/latest/installation.html)。

您需要 Python 3.8 或更高版本。通过 PyPI 安装 Flax：

```
pip install flax
```

若需升级至 Flax 最新版本：

```
pip install --upgrade git+https://github.com/google/flax.git
```

若需安装某些示例所需的扩展依赖（如 `matplotlib` 等）：

```bash
pip install "flax[all]"
```

## Flax 代码范式

我们通过 Flax API 提供了三个经典示例：简易多层感知机（MLP）、卷积神经网络（CNN）与自编码器（Auto-encoder）。

若需深入了解 `Module` 抽象，请查阅我们的[官方文档](https://flax.readthedocs.io/)，以及[模块抽象全面入门指南](https://github.com/google/flax/blob/main/docs/linen_intro.ipynb)。有关最佳实践的更多具体演示，请参阅[实战指南](https://flax.readthedocs.io/en/latest/guides/index.html)和[开发者笔记](https://flax.readthedocs.io/en/latest/developer_notes/index.html)。

多层感知机（MLP）示例：

```py
class MLP(nnx.Module):
  def __init__(self, din: int, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(din, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(rate=0.1, rngs=rngs)
    self.bn = nnx.BatchNorm(dmid, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dout, rngs=rngs)

  def __call__(self, x: jax.Array):
    x = nnx.gelu(self.dropout(self.bn(self.linear1(x))))
    return self.linear2(x)
```

卷积神经网络（CNN）示例：

```py
class CNN(nnx.Module):
  def __init__(self, *, rngs: nnx.Rngs):
    self.conv1 = nnx.Conv(1, 32, kernel_size=(3, 3), rngs=rngs)
    self.conv2 = nnx.Conv(32, 64, kernel_size=(3, 3), rngs=rngs)
    self.avg_pool = partial(nnx.avg_pool, window_shape=(2, 2), strides=(2, 2))
    self.linear1 = nnx.Linear(3136, 256, rngs=rngs)
    self.linear2 = nnx.Linear(256, 10, rngs=rngs)

  def __call__(self, x):
    x = self.avg_pool(nnx.relu(self.conv1(x)))
    x = self.avg_pool(nnx.relu(self.conv2(x)))
    x = x.reshape(x.shape[0], -1)  # flatten
    x = nnx.relu(self.linear1(x))
    x = self.linear2(x)
    return x
```

自编码器（Auto-encoder）示例：

```py
Encoder = lambda rngs: nnx.Linear(2, 10, rngs=rngs)
Decoder = lambda rngs: nnx.Linear(10, 2, rngs=rngs)

class AutoEncoder(nnx.Module):
  def __init__(self, rngs):
    self.encoder = Encoder(rngs)
    self.decoder = Decoder(rngs)

  def __call__(self, x) -> jax.Array:
    return self.decoder(self.encoder(x))

  def encode(self, x) -> jax.Array:
    return self.encoder(x)
```

## 引用 Flax

如需引用本仓库：

```
@software{flax2020github,
  author = {Jonathan Heek and Anselm Levskaya and Avital Oliver and Marvin Ritter and Bertrand Rondepierre and Andreas Steiner and Marc van {Z}ee},
  title = {{F}lax: A neural network library and ecosystem for {JAX}},
  url = {http://github.com/google/flax},
  version = {0.12.10},
  year = {2026},
}
```

在上述 BibTeX 条目中，作者姓名按字母顺序排列，版本号对应于 [flax/version.py](https://github.com/google/flax/blob/main/flax/version.py)，年份对应于该项目开源发布的年份。

## 注意事项

Flax 是由 Google DeepMind 专属团队维护的开源项目，但不是 Google 官方正式产品。

---

> 💡 **文档维护说明**：本中文文档由社区志愿者（[@JasonYeYuhe](https://github.com/JasonYeYuhe)）翻译维护，最后同步更新于 2026年09月29日。如发现内容与官方英文原版存在差异或新特性滞后，欢迎提交 PR 共同完善！
