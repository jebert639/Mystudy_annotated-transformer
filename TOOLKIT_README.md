# Transformer 工具箱使用说明

本目录新增了一套**高层封装**，让训练和推理更省事。**原有的文件一律没有改动**，新代码都是在原有 `train_runner.py` / `train_utils.py` / `transformer_model.py` 外面套了一层。

## 新增文件一览

| 文件 | 作用 |
|---|---|
| `transformer_api.py` | 核心封装：一行训练 `train(...)`、模型列表 `list_models()`、一行推理 `infer(...)` |
| `model_gui.py` | 推理小界面（Tkinter，Python 自带，**无需安装任何东西**） |
| `demo_train.py` | 最小训练脚本示例，直接运行 = 冒烟测试 |
| `launch_gui.bat` | 双击启动推理界面（自动用 annotated-transformer 环境） |
| `TOOLKIT_README.md` | 本说明 |

## 一、一行训练

以前：

```python
from train_runner import train
train(data_path="the-verdict.txt", output_dir="output", d_model=512, n=6, h=8, ...)
```

现在（新脚本里这样写即可）：

```python
from transformer_api import train

train()                                            # 全默认, 训练 the-verdict.txt
train(preset="tiny")                               # 冒烟测试: 最小模型 2 个 epoch
train(preset="fast")                               # 快速实验: 256维/4层/30 epoch
train(preset="fast", d_model=384, num_epochs=50)   # 预设基础上想改哪个写哪个
train(d_model=512, n=6, h=8, d_ff=2048,
      warmup_steps=4000, num_epochs=200)           # 论文标准配置
```

规则：
- **没写的参数沿用原 `train_runner.train` 的默认值**；写了的就覆盖（包括覆盖预设）。
- `data_path` 不写默认自动找项目根的 `the-verdict.txt`。
- `output_dir` 不写默认 `output`，已存在会自动变成 `output_2`、`output_3`…（原逻辑不变）。
- 额外支持 `seed=42`（可复现）、`device="cpu"`（强制 CPU）。

### 预设对照

| 预设 | d_model | N | h | d_ff | batch | seq_len | epochs | warmup | 用途 |
|---|---|---|---|---|---|---|---|---|---|
| `tiny` | 64 | 2 | 2 | 128 | 32 | 32 | 2 | 20 | 冒烟测试整条链路 |
| `fast` | 256 | 4 | 4 | 1024 | 64 | 48 | 30 | 200 | 日常快速实验 |
| `paper` | 512 | 6 | 8 | 2048 | 128 | 64 | 200 | 4000 | 论文标准配置 |

也可以不用预设，全部自己传。

## 二、看有哪些模型

```bash
python transformer_api.py list
```

自动扫描项目里所有 `output*/` 目录下的 `.pt` 文件，显示每个模型的类型（best/final）、epoch、val_loss、结构配置、大小、修改时间。Python 里用 `list_models()`，返回结构化列表。

## 三、一行推理

```bash
python transformer_api.py infer "I had always thought" --model output/best_model.pt
```

Python 里：

```python
from transformer_api import infer, load_model, generate

infer("I had always thought")            # model 不写 = 自动选 val_loss 最低的 best 模型
infer("I had always thought", model="output_2/best_model.pt", temperature=0.7)

# 反复生成时, 先加载一次再生成 (GUI 就是这么做的):
pack = load_model("output/best_model.pt")
generate(pack, "I had always thought", max_len=100, temperature=0.7)
generate(pack, "The judge", max_len=100, temperature=0.7)
```

`temperature`：0 = 纯贪心（容易复读循环），0.7 = 推荐，1.0 = 标准采样。

## 四、推理界面

三种启动方式任选：

1. 双击 `launch_gui.bat`
2. `python model_gui.py`
3. `python transformer_api.py gui`

界面功能：
- **左侧列表**：自动列出所有模型，含类型/epoch/val_loss/结构/大小；「刷新」重新扫描，「浏览」可选任意位置的 `.pt`。
- **双击模型** = 加载（后台线程加载，界面不卡死）。
- **右侧**：模型详情（参数量、设备、训练到第几轮）→ 输入前缀 → 调温度和长度 → 「生成补全」。
- 结果区保留历史记录，每条带模型名、温度、耗时。
- 「查看训练曲线」用系统看图工具打开该模型目录下的 `loss_curve.png`。

## 五、常见问题

**用哪个 Python 环境？**
训练/推理都要用 conda 的 `annotated-transformer` 环境（torch 1.11 + tiktoken）：
```bash
conda activate annotated-transformer
```
`launch_gui.bat` 已自动指向这个环境。

**命令行训练怎么写？**
```bash
python transformer_api.py train preset=fast num_epochs=30
python transformer_api.py train d_model=256 seq_len=48
```
`key=value` 自动识别 int/float/bool，写错参数名会直接报错提示。

**输出去哪了？**
每个输出目录里有：`best_model.pt`（按验证 loss 保存的最佳模型）、`final_model.pt`、`train.log`、`loss_curve.png`、`lr_curve.png`、`loss_lr_combined.png`。界面里看到的 val_loss 就来自 checkpoint 里记录的值。
