# fairseq-light

`fairseq-light` 是一个面向推理（inference）的 `fairseq` 轻量化兼容层：发行包名为 `fairseq-light`，但 Python 包名仍为 `fairseq`，用于尽量兼容第三方项目里既有的 `import fairseq...` 代码路径。

## 为什么需要它

在很多推理/部署场景里，我们往往只是“被动依赖” `fairseq` 的模型/模块实现（例如一些模型结构、attention/transformer 组件等），但 upstream `fairseq` 本身是偏训练框架的工程：

- 导入链路长、模块间依赖复杂，容易出现“导入即触发大量子模块/可选依赖”的问题
- 包含大量与推理无关的功能与依赖（训练 CLI、任务/criterion/optimizer、数据处理、分布式训练等），增加部署成本与冲突概率

`fairseq-light` 的目标是在推理环境里提供一个更轻量、更容易安装/导入的 `fairseq` 子集，提升兼容性与可控性。

## 和 upstream fairseq 的关系

- 本项目基于 `pytorch/fairseq` 代码裁剪/整理而来（非官方、非上游替代品）。
- 目标是“推理优先的兼容层”，而不是完整复刻 upstream `fairseq` 的全部能力。
- 如果你需要训练、完整的数据与任务体系、命令行工具（如 `fairseq-train`/`fairseq-generate`）等，请使用 upstream `fairseq`。

## 安装

```bash
pip install fairseq-light
```

`fairseq-light` 不强制依赖 PyTorch；请按你的运行环境自行安装：

```bash
pip install torch
```

> 注意：由于本项目同样占用 `fairseq` 这个 Python 包名，不建议与 upstream `fairseq` 在同一环境同时安装。  
> 如已安装 upstream 版本，请先执行 `pip uninstall fairseq` 再安装 `fairseq-light`。

## 快速验证

```python
import fairseq
from fairseq.modules import MultiheadAttention, TransformerEncoderLayer
```

## 兼容性说明

- 这里只覆盖推理常用的 `fairseq` 子集；某些 `fairseq.*` 导入路径/功能可能未包含或需要额外依赖。
- 部分模型/接口可能依赖额外第三方库（如 `sentencepiece`、`spacy`、音频相关库等），这些依赖不会默认安装；请按需自行安装。
- 遇到缺失的模块或导入报错：欢迎提 issue/PR，并附上最小复现的 import 路径与报错信息，便于补齐兼容面。

## 版本策略

`fairseq-light` 的版本号与 upstream `fairseq` 没有严格一一对应关系；以本项目发布版本为准。
