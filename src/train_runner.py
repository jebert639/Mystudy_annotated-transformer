"""
训练运行器 - 封装完整的训练/推理流程
提供 train() 和 infer() 两个顶层函数

超参数默认值遵循 Transformer 原论文 (Attention Is All You Need):
  d_model=512, N=6, h=8, d_ff=2048, warmup=4000, label_smooth=0.1
"""

import os
import time
import random
import logging
import torch
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader

from transformer_model import make_model
from train_utils import (
    BPETokenizer,
    TextDataset,
    collate_batch,
    LabelSmoothing,
    SimpleLossCompute,
    rate,
    run_epoch,
    complete_sentence,
)
from plot_utils import plot_loss_curve, plot_lr_curve, plot_loss_lr_combined


def _make_output_dir(base_dir):
    if os.path.exists(base_dir):
        idx = 2
        while os.path.exists(f"{base_dir}_{idx}"):
            idx += 1
        return f"{base_dir}_{idx}"
    return base_dir


def _setup_logger(output_dir):
    log_path = os.path.join(output_dir, "train.log")
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fh = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(message)s", datefmt="%Y-%m-%d %H:%M:%S"))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(ch)
    return logger


def train(
    data_path,
    output_dir="output",
    d_model=512,
    n=6,
    h=8,
    d_ff=2048,
    dropout=0.3,
    batch_size=128,
    seq_len=64,
    num_epochs=200,
    warmup_steps=400,
    label_smooth=0.1,
    stride=4,
    prompt_ratio=0.5,
    val_ratio=0.2,
    weight_decay=0.0,
    overwrite=False,
):
    """
    完整训练流程, 默认参数遵循 Transformer 原论文

    论文标准配置:
        d_model=512, N=6, h=8, d_ff=2048, warmup=4000, label_smooth=0.1

    数据划分:
        原始文本按字符顺序切分, 前80%为训练集, 后20%为验证集
        best_model 按 验证集loss 保存

    返回:
        dict: 包含 best_val_loss, final_train_loss, total_params, output_dir, ...
    """
    base_dir = os.path.abspath(output_dir)
    if overwrite:
        out_dir = base_dir
    else:
        out_dir = _make_output_dir(base_dir)
    os.makedirs(out_dir, exist_ok=True)

    logger = _setup_logger(out_dir)
    log = logger.info

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log(f"使用设备: {device}")
    log(f"\n========== 超参数配置 (遵循 Transformer 原论文) ==========")
    log(f"  d_model       = {d_model}    (论文: 512)")
    log(f"  N (层数)      = {n}    (论文: 6)")
    log(f"  h (头数)      = {h}    (论文: 8)")
    log(f"  d_ff          = {d_ff}   (论文: 2048)")
    log(f"  dropout       = {dropout}  (论文: 0.1)")
    log(f"  warmup_steps  = {warmup_steps}   (论文: 4000, 小数据集建议: 总步数/10)")
    log(f"  label_smooth  = {label_smooth}  (论文: 0.1)")
    log(f"  batch_size    = {batch_size}")
    log(f"  seq_len       = {seq_len}")
    log(f"  num_epochs    = {num_epochs}")
    log(f"  stride        = {stride}")
    log(f"  prompt_ratio  = {prompt_ratio}")
    log(f"  val_ratio     = {val_ratio}  (验证集比例)")
    log(f"  weight_decay  = {weight_decay}  (权重衰减)")
    log(f"=========================================================\n")

    with open(data_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    log(f"数据集大小: {len(raw_text)} 字符")

    tokenizer = BPETokenizer(raw_text)
    vocab_size = tokenizer.vocab_size
    log(f"词表大小: {vocab_size} (gpt2 BPE: 50257 + 3 特殊 token)")

    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    random.seed(42)
    random.shuffle(paragraphs)
    split_idx = int(len(paragraphs) * (1 - val_ratio))
    train_paragraphs = paragraphs[:split_idx]
    val_paragraphs = paragraphs[split_idx:]
    train_text = "\n\n".join(train_paragraphs)
    val_text = "\n\n".join(val_paragraphs)
    log(f"段落总数: {len(paragraphs)}")
    log(f"训练集: {len(train_paragraphs)} 段落, {len(train_text)} 字符 ({(1 - val_ratio) * 100:.0f}%)")
    log(f"验证集: {len(val_paragraphs)} 段落, {len(val_text)} 字符 ({val_ratio * 100:.0f}%)")

    model = make_model(vocab_size, vocab_size, N=n, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    log(f"模型参数量: {total_params:,}")

    train_dataset = TextDataset(train_text, tokenizer, seq_len, stride=stride, prompt_ratio=prompt_ratio)
    val_dataset = TextDataset(val_text, tokenizer, seq_len, stride=stride, prompt_ratio=prompt_ratio)
    log(f"训练样本数: {len(train_dataset)}")
    log(f"验证样本数: {len(val_dataset)}")

    collate_fn = lambda batch: collate_batch(batch, tokenizer.pad_id)
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_fn
    )

    criterion = LabelSmoothing(size=vocab_size, padding_idx=tokenizer.pad_id, smoothing=label_smooth)
    loss_compute = SimpleLossCompute(model.generator, criterion)

    optimizer = torch.optim.Adam(
        model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9, weight_decay=weight_decay
    )
    lr_scheduler = LambdaLR(
        optimizer=optimizer,
        lr_lambda=lambda step: rate(step, model_size=d_model, factor=1.0, warmup=warmup_steps),
    )

    log(f"\n开始训练: {num_epochs} epochs, batch_size={batch_size}, seq_len={seq_len}")
    log("-" * 60)

    best_val_loss = float("inf")
    train_losses = []
    val_losses = []
    step_lrs = []

    start_time = time.time()
    for epoch in range(num_epochs):
        model.train()
        train_loss = run_epoch(
            train_loader, model, loss_compute, optimizer, lr_scheduler, mode="train"
        )
        train_losses.append(float(train_loss))
        step_lrs.append(optimizer.param_groups[0]["lr"])

        model.eval()
        with torch.no_grad():
            val_loss = run_epoch(
                val_loader, model, loss_compute, optimizer, lr_scheduler, mode="eval"
            )
        val_losses.append(float(val_loss))

        log(
            f"Epoch {epoch + 1:3d}/{num_epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"LR: {optimizer.param_groups[0]['lr']:.2e}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            ckpt_path = os.path.join(out_dir, "best_model.pt")
            torch.save({
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "config": {
                    "d_model": d_model, "N": n, "h": h, "d_ff": d_ff,
                    "dropout": dropout, "vocab_size": vocab_size,
                },
            }, ckpt_path)
            log(f"  -> 保存最佳模型 (val_loss={best_val_loss:.4f})")

    total_time = time.time() - start_time

    final_ckpt_path = os.path.join(out_dir, "final_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "epoch": num_epochs,
        "train_loss": train_losses[-1],
        "val_loss": val_losses[-1],
        "config": {
            "d_model": d_model, "N": n, "h": h, "d_ff": d_ff,
            "dropout": dropout, "vocab_size": vocab_size,
        },
    }, final_ckpt_path)

    plot_loss_curve(train_losses, val_losses, out_dir)
    plot_lr_curve(step_lrs, out_dir)
    plot_loss_lr_combined(train_losses, val_losses, step_lrs, out_dir)

    log("-" * 60)
    log(f"训练完成! 总耗时: {total_time / 60:.1f} 分钟")
    log(f"最佳验证 Loss: {best_val_loss:.4f}")
    log(f"最终训练 Loss: {train_losses[-1]:.4f}")
    log(f"最终验证 Loss: {val_losses[-1]:.4f}")
    log(f"模型参数量: {total_params:,}")
    log(f"\n输出文件夹: {out_dir}")
    log(f"  - best_model.pt    (最佳模型, 按验证集loss保存)")
    log(f"  - final_model.pt   (最终模型)")
    log(f"  - train.log        (训练日志)")
    log(f"  - loss_curve.png   (训练+验证损失曲线)")
    log(f"  - lr_curve.png     (学习率曲线)")
    log(f"  - loss_lr_combined.png (Loss+LR合并图)")

    return {
        "best_val_loss": best_val_loss,
        "final_train_loss": train_losses[-1],
        "final_val_loss": val_losses[-1],
        "total_params": total_params,
        "output_dir": out_dir,
        "train_losses": train_losses,
        "val_losses": val_losses,
        "step_lrs": step_lrs,
    }


def infer(
    prompt,
    model_path="output/best_model.pt",
    max_len=200,
    temperature=0.7,
):
    """
    推理: 输入前缀文本, 模型自动补全后续内容

    返回:
        str: prompt + 补全后的文本
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(model_path, map_location=device)
    config = checkpoint.get("config", {})

    d_model = config.get("d_model", 512)
    n = config.get("N", 6)
    h = config.get("h", 8)
    d_ff = config.get("d_ff", 2048)
    dropout = config.get("dropout", 0.1)
    vocab_size = config.get("vocab_size", 100259)

    tokenizer = BPETokenizer()

    model = make_model(vocab_size, vocab_size, N=n, d_model=d_model, d_ff=d_ff, h=h, dropout=dropout)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)

    result = complete_sentence(model, tokenizer, prompt, max_len=max_len, device=device, temperature=temperature)
    return result