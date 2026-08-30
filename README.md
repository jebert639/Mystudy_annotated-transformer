<p align="center">
  <img src="tansformer.png" alt="Transformer" width="400"/>
</p>

# Annotated Transformer 中文实践

> 从逐行实现 Transformer 到句子补全 + 推理工具箱

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

基于哈佛 NLP 的 **The Annotated Transformer**（[Attention Is All You Need](https://arxiv.org/abs/1706.03762) 逐行注释版），本项目做了三件事：

1. **从零实现完整 Transformer** — 代码与论文 Figure 1 逐块对应，附中文注释
2. **英文句子补全（续写）任务** — 自回归生成，支持 BPE 分词、温度采样
3. **一行式训练 API + 零依赖桌面推理界面** — 训练只需 `train(参数=值, ...)`

---

## 快速开始

```bash
# 1. 创建环境 (Python 3.9)
conda create -n annotated-transformer python=3.9 -y
conda activate annotated-transformer

# 2. 安装依赖
pip install -r requirements.txt

# 3. 冒烟测试 (~2 分钟)
python demo_train.py

# 4. 启动推理界面
python model_gui.py
```

## 详细介绍

👉 **[查看完整中文介绍 →](PROJECT_INTRO.md)**

包含：实验环境、文件结构总览、训练/推理/界面使用指南、预设配置对照表、学习笔记与数据流说明。

## 工具文档

👉 **[查看工具箱使用说明 →](TOOLKIT_README.md)**

包含：一行训练 API、模型列表查看、一行推理、推理界面操作指南。

## 原版参考

本仓库基于 [harvardnlp/annotated-transformer](https://github.com/harvardnlp/annotated-transformer)（MIT License），保留了原版 notebook 构建工具：

```bash
make notebook   # 从 .py 生成 .ipynb
make html       # 生成 html 版本
make black      # 自动格式化代码
make flake      # PEP8 检查
```

## License

MIT License。详见 [LICENSE](LICENSE)。