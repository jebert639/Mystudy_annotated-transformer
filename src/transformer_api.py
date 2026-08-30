"""
transformer_api.py - Transformer 训练/推理一站式封装 (高层 API)
================================================================

在不改动原有代码 (train_runner.py / train_utils.py / transformer_model.py) 的
前提下, 提供更好用的一站式接口:

    1. train(...)      一行启动训练: train(参数名=值, ...)
    2. list_models()   自动扫描项目里的 outputs/ 目录, 列出所有可用模型
    3. infer(...)      一行推理, model 可以是 路径 / 序号 / None(自动选最优)
    4. load_model()    加载模型供反复使用 (GUI 用的底层接口)

用法示例:

    from transformer_api import train, infer, list_models

    train()                                            # 全部默认, 训练 data/the-verdict.txt
    train(preset="tiny")                               # 冒烟测试: 最小模型 2 个 epoch
    train(name="big512_v2", preset="fast")             # 给模型起名 -> outputs/big512_v2
    train(preset="fast")                               # 快速实验配置
    train(preset="fast", d_model=384, num_epochs=50)   # 预设基础上覆盖若干参数
    train(d_model=512, n=6, h=8, d_ff=2048, warmup_steps=4000, num_epochs=200)  # 论文配置

    list_models()                                      # 打印模型列表
    infer("I had always thought")                      # 自动加载 val_loss 最低的 best 模型
    infer("I had always thought", model="outputs/output_2/best_model.pt")

命令行 (在项目根目录执行):

    python src/transformer_api.py list
    python src/transformer_api.py train preset=fast num_epochs=30
    python src/transformer_api.py infer "I had always thought" --model outputs/output/best_model.pt
    python src/transformer_api.py gui      # 启动推理界面

注意: 数据集段落划分沿用 train_runner 内部固定的 random.seed(42),
     这里的 seed 只控制模型初始化/采样等 torch 侧随机性。
"""

import os
import re
import time
import random
import argparse

import torch

from transformer_model import make_model
from train_utils import BPETokenizer, complete_sentence

# 注意: 不要在模块顶层 import train_runner !
# train_runner -> plot_utils -> matplotlib -> freetype/harfbuzz,
# 在未激活 conda 的环境下(如直接双击 bat), matplotlib 会按名字搜到
# H:\msys2\mingw64\bin\libharfbuzz-0.dll 并因版本不匹配弹窗崩溃。
# 惰性导入后, list/infer/GUI 完全不碰 matplotlib, 只有 train() 才加载。

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/ 的上一级 = 项目根
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
OUTPUTS_DIR = os.path.join(PROJECT_ROOT, "outputs")


# ============================================================
# 预设 (preset)
# ============================================================
# 只列出想覆盖的参数, 其余参数沿用 train_runner.train 的默认值

PRESETS = {
    # 冒烟测试: 30 秒~几分钟跑完整条链路 (建模型/训练/存档/绘图), 确认环境没问题
    "tiny": dict(
        d_model=64, n=2, h=2, d_ff=128,
        batch_size=32, seq_len=32, num_epochs=2,
        warmup_steps=20, stride=32, dropout=0.1,
    ),
    # 日常快速实验: 小模型 + 少 epoch, 快速看趋势
    "fast": dict(
        d_model=256, n=4, h=4, d_ff=1024,
        batch_size=64, seq_len=48, num_epochs=30,
        warmup_steps=200, stride=8, dropout=0.2,
    ),
    # 论文标准配置 (Attention Is All You Need): d_model=512, N=6, h=8, d_ff=2048, warmup=4000
    "paper": dict(
        d_model=512, n=6, h=8, d_ff=2048,
        batch_size=128, seq_len=64, num_epochs=200,
        warmup_steps=4000, stride=4, dropout=0.1,
    ),
}


def set_seed(seed=42):
    "统一设置随机种子 (python + torch + cuda), 保证实验可复现"
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_data_path(path):
    "定位训练数据文件: 绝对路径直接用; 相对路径先按当前目录找, 再按 data/ 和项目根找"
    if os.path.isabs(path):
        if not os.path.exists(path):
            raise FileNotFoundError(f"训练数据不存在: {path}")
        return path
    if os.path.exists(path):
        return os.path.abspath(path)
    for alt in (os.path.join(DATA_DIR, path), os.path.join(PROJECT_ROOT, path)):
        if os.path.exists(alt):
            return alt
    raise FileNotFoundError(
        f"训练数据不存在: {path} (也尝试了 {os.path.join(DATA_DIR, path)})"
    )


# ============================================================
# 训练
# ============================================================

def train(preset=None, name=None, data_path=None, output_dir=None, seed=42, device=None, **hyper):
    """
    一行启动训练, 写脚本时直接: train(参数名=值, 参数名=值, ...)

    参数:
        preset      预设名: "tiny" / "fast" / "paper", 可省略
        name        模型名: 作为 outputs/ 下的子目录名 (如 name="big512_v2"
                    -> outputs/big512_v2)。不指定则默认 outputs/output 并
                    自动递增 output_2、output_3...; 已存在同名目录会自动加
                    _2 后缀, 想覆盖请传 overwrite=True
        data_path   训练文本路径, 默认自动找 data/the-verdict.txt
        output_dir  输出目录 (与 name 二选一, 同时给出时 output_dir 优先);
                    相对名自动落到 outputs/ 下, 绝对路径按原样使用
        seed        随机种子 (默认 42)
        device      None=自动 (有CUDA用CUDA); 传 "cpu" 可强制用CPU
        **hyper     其余全部直接透传给 train_runner.train, 例如:
                    d_model, n, h, d_ff, dropout, batch_size, seq_len,
                    num_epochs, warmup_steps, label_smooth, stride,
                    prompt_ratio, val_ratio, weight_decay, overwrite

    返回:
        dict: 包含 best_val_loss, output_dir, train_losses, val_losses, ...

    示例:
        train()                                # 默认配置
        train(preset="fast", d_model=384)      # 预设 + 覆盖
    """
    if preset is not None:
        if preset not in PRESETS:
            raise ValueError(
                f"未知预设 '{preset}', 可选: {sorted(PRESETS)} "
                f"(也可以不用 preset, 直接传参数, 如 train(d_model=256))"
            )
        cfg = dict(PRESETS[preset])
    else:
        cfg = {}

    # 显式传入的参数覆盖预设
    cfg.update(hyper)

    # 模型命名: name 指定 outputs/ 下的子目录名; 未指定则默认 output (自动递增)
    if output_dir is None:
        output_dir = name if name is not None else "output"

    # 所有训练产物统一放到项目根的 outputs/ 目录下
    if not os.path.isabs(output_dir):
        output_dir = os.path.join(OUTPUTS_DIR, output_dir)

    if data_path is None:
        data_path = os.path.join(DATA_DIR, "the-verdict.txt")
    data_path = _resolve_data_path(data_path)

    if device == "cpu":
        # 让 train_runner 里 "cuda if available" 的判断失效, 强制走 CPU
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    elif device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device='cuda' 但当前机器没有可用的 CUDA")

    set_seed(seed)

    # 惰性导入: 只有真正训练才加载 train_runner/matplotlib (见文件头注释)
    # Windows conda 环境下 torch 与 matplotlib 各自带了一份 OpenMP 运行时
    # (libiomp5md.dll), 同时初始化会触发 OMP Error #15 直接崩溃,
    # 这里按官方 hint 设置 KMP_DUPLICATE_LIB_OK 规避。
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    from train_runner import train as _train

    tag = f"preset={preset}" if preset else "自定义参数"
    print(f"[transformer_api] 开始训练 ({tag}, seed={seed})")
    if cfg:
        print(f"[transformer_api] 本次指定: {cfg}")

    return _train(data_path=data_path, output_dir=output_dir, **cfg)


# ============================================================
# 模型发现与管理
# ============================================================

def _natural_key(s):
    "自然排序: output_2 排在 output_10 前面"
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", s)]


def _torch_load(path):
    "兼容不同 torch 版本的 checkpoint 读取"
    try:
        return torch.load(path, map_location="cpu")
    except TypeError:  # torch>=2.6 默认 weights_only=True, 老格式会报错
        return torch.load(path, map_location="cpu", weights_only=False)


def list_models(root=PROJECT_ROOT, verbose=True):
    """
    扫描项目目录下所有 output*/ 文件夹里的 .pt 模型, 读取元信息

    返回:
        list[dict], 每项包含:
            path      相对路径 (如 output/best_model.pt)
            abs_path  绝对路径
            kind      best / final / other
            epoch, val_loss, train_loss
            config    模型结构 (d_model, N, h, d_ff, dropout, vocab_size)
            size_mb   文件大小
            mtime     修改时间
    """
    import glob

    models = []
    pt_files = glob.glob(os.path.join(root, "outputs", "*", "*.pt"))
    for pt in sorted(pt_files, key=lambda p: (_natural_key(os.path.basename(os.path.dirname(p))), os.path.basename(p))):
        try:
            ck = _torch_load(pt)
        except Exception as e:
            if verbose:
                print(f"[transformer_api] 跳过无法读取的文件 {pt}: {e}")
            continue

        rel = os.path.relpath(pt, root)
        name = os.path.basename(pt)
        kind = "best" if name.startswith("best") else ("final" if name.startswith("final") else "other")
        val_loss = ck.get("val_loss")
        train_loss = ck.get("train_loss")
        models.append({
            "path": rel,
            "abs_path": pt,
            "kind": kind,
            "epoch": ck.get("epoch", "?"),
            "val_loss": float(val_loss) if val_loss is not None else None,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "config": ck.get("config", {}),
            "size_mb": os.path.getsize(pt) / 1024 / 1024,
            "mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(pt))),
        })

    if verbose:
        print_models(models)
    return models


def print_models(models):
    "把 list_models 的结果打印成对齐表格"
    if not models:
        print("没有找到任何模型 (项目目录下没有 output*/*.pt)")
        return
    header = f"{'模型文件':<26} {'类型':<6} {'Epoch':>5} {'ValLoss':>8} {'配置':<32} {'大小':>8}  {'修改时间'}"
    print(header)
    print("-" * len(header))
    for m in models:
        c = m["config"]
        cfg = (f"d_model={c.get('d_model','?')} N={c.get('N','?')} h={c.get('h','?')} "
               f"d_ff={c.get('d_ff','?')}") if c else "-"
        val = f"{m['val_loss']:.3f}" if m["val_loss"] is not None else "-"
        print(
            f"{m['path']:<26} {m['kind']:<6} {str(m['epoch']):>5} {val:>8} "
            f"{cfg:<32} {m['size_mb']:>7.1f}M  {m['mtime']}"
        )


def _resolve_model(model=None, root=PROJECT_ROOT):
    """
    把 model 参数解析成 checkpoint 绝对路径:
        None  -> 自动选 val_loss 最低的 best 模型 (没有 best 就选任意最新)
        int   -> list_models() 顺序中的第 model 个 (从 0 开始)
        str   -> 路径 (相对路径按项目根解析)
    """
    models = list_models(root, verbose=False)
    if model is None:
        if not models:
            raise FileNotFoundError(
                "没有找到任何模型文件。请先训练一个模型 (train(...)), "
                "或把 .pt 文件放到 output*/ 目录下"
            )
        best = [m for m in models if m["kind"] == "best" and m["val_loss"] is not None]
        pool = sorted(best or models, key=lambda m: float("inf") if m["val_loss"] is None else m["val_loss"])
        return pool[0]["abs_path"]

    if isinstance(model, int):
        if not models:
            raise FileNotFoundError("没有找到任何模型文件")
        return models[model]["abs_path"]

    if isinstance(model, str):
        path = model
        if not os.path.isabs(path):
            cand = os.path.join(root, path)
            path = cand if os.path.exists(cand) else os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"模型文件不存在: {model} (解析为 {path})")
        return path

    raise TypeError(f"model 参数类型不支持: {type(model)}, 应为 None / int / str")


# ============================================================
# 推理
# ============================================================

def load_model(model=None, device=None, root=PROJECT_ROOT):
    """
    加载模型, 返回可复用的"模型包" (反复生成时不要每次重新加载)

    参数:
        model   None=自动选最优 / 文件路径 / list_models 里的序号
        device  None=自动; "cpu" / "cuda"

    返回:
        dict: {"model", "tokenizer", "device", "info", "path"}
    """
    path = _resolve_model(model, root)
    ck = _torch_load(path)
    config = ck.get("config", {})

    d_model = config.get("d_model", 512)
    n = config.get("N", 6)
    h = config.get("h", 8)
    d_ff = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    vocab_size = config.get("vocab_size", 100259)

    net = make_model(vocab_size, vocab_size, N=n, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
    net.load_state_dict(ck["model_state_dict"])

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    net = net.to(device).eval()

    info = {
        "path": os.path.relpath(path, root),
        "epoch": ck.get("epoch", "?"),
        "val_loss": float(ck["val_loss"]) if ck.get("val_loss") is not None else None,
        "train_loss": float(ck["train_loss"]) if ck.get("train_loss") is not None else None,
        "total_params": sum(p.numel() for p in net.parameters()),
        "config": config,
    }
    print(f"[transformer_api] 已加载 {info['path']} "
          f"(epoch={info['epoch']}, val_loss={info['val_loss']}, 参数量={info['total_params']:,}, device={device})")

    return {
        "model": net,
        "tokenizer": BPETokenizer(),
        "device": device,
        "info": info,
        "path": path,
    }


def generate(pack, prompt, max_len=200, temperature=0.7):
    """
    用已加载的模型包做补全 (GUI 场景: 一次加载, 反复生成)

    参数:
        pack         load_model() 的返回值
        prompt       前缀文本
        max_len      最多生成的 token 数
        temperature  0=纯贪心(容易循环), 0.7=推荐, 1.0=标准采样
    """
    return complete_sentence(
        pack["model"], pack["tokenizer"], prompt,
        max_len=max_len, device=pack["device"], temperature=temperature,
    )


def infer(prompt, model=None, max_len=200, temperature=0.7, device=None, root=PROJECT_ROOT):
    """
    一行推理: infer("I had always thought", model="output/best_model.pt")

    model=None 时自动加载 val_loss 最低的 best 模型。
    每次调用都会重新加载模型; 需要反复生成请用 load_model() + generate()。
    """
    pack = load_model(model, device=device, root=root)
    return generate(pack, prompt, max_len=max_len, temperature=temperature)


# ============================================================
# 命令行
# ============================================================

def _coerce(v):
    "把命令行里的 key=value 值转成合适类型: 256->int, 0.3->float, true->bool"
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def main():
    parser = argparse.ArgumentParser(description="Transformer 工具箱 (transformer_api)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="列出项目里所有可用的模型")

    p_train = sub.add_parser("train", help="训练: train preset=fast num_epochs=30")
    p_train.add_argument("overrides", nargs="*", help="key=value 形式的参数覆盖, 如 d_model=256")

    p_infer = sub.add_parser("infer", help="推理: infer \"前缀文本\" --model 路径")
    p_infer.add_argument("prompt", help="前缀文本")
    p_infer.add_argument("--model", default=None, help="模型路径 (默认自动选 val_loss 最低的 best)")
    p_infer.add_argument("--max_len", type=int, default=200)
    p_infer.add_argument("--temperature", type=float, default=0.7)

    sub.add_parser("gui", help="启动推理界面 (model_gui.py)")

    args = parser.parse_args()

    if args.cmd == "list":
        list_models(verbose=True)

    elif args.cmd == "train":
        kwargs = {}
        for item in args.overrides:
            if "=" not in item:
                raise SystemExit(f"参数格式应为 key=value, 收到: {item}")
            k, v = item.split("=", 1)
            kwargs[k] = _coerce(v)
        result = train(**kwargs)
        print(f"\n训练完成! 输出目录: {result['output_dir']}")
        print(f"最佳验证 loss: {result['best_val_loss']:.4f}")

    elif args.cmd == "infer":
        result = infer(args.prompt, model=args.model, max_len=args.max_len, temperature=args.temperature)
        print(f"输入: {args.prompt}")
        print(f"补全: {result}")

    elif args.cmd == "gui":
        import model_gui
        model_gui.main()


if __name__ == "__main__":
    main()
