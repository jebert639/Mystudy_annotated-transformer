"""
训练可视化工具
绘制损失曲线和学习率曲线, 保存到输出文件夹
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_loss_curve(train_losses, val_losses, output_dir):
    """
    绘制训练+验证损失曲线并保存
    """
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, "b-", linewidth=1.5, label="Train Loss")
    plt.plot(epochs, val_losses, "r-", linewidth=1.5, label="Val Loss")
    plt.xlabel("Epoch", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training & Validation Loss", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "loss_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_lr_curve(lrs, output_dir):
    """
    绘制学习率变化曲线并保存
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(lrs)), lrs, "r-", linewidth=1.0, label="Learning Rate")
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("LR", fontsize=12)
    plt.title("Learning Rate Schedule", fontsize=14)
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    save_path = os.path.join(output_dir, "lr_curve.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path


def plot_loss_lr_combined(train_losses, val_losses, lrs, output_dir):
    """
    双Y轴: 左边Loss(训练+验证), 右边LR, 合在一张图里
    """
    fig, ax1 = plt.subplots(figsize=(12, 6))

    color_train = "tab:blue"
    color_val = "tab:red"
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    epochs = range(1, len(train_losses) + 1)
    ax1.plot(epochs, train_losses, color=color_train, linewidth=1.5, label="Train Loss")
    ax1.plot(epochs, val_losses, color=color_val, linewidth=1.5, label="Val Loss")
    ax1.tick_params(axis="y")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper left", fontsize=11)

    ax2 = ax1.twinx()
    color_lr = "tab:green"
    ax2.set_ylabel("Learning Rate", color=color_lr, fontsize=12)
    epochs_lrs = lrs[:: max(1, len(lrs) // len(train_losses))][: len(train_losses)]
    ax2.plot(range(1, len(epochs_lrs) + 1), epochs_lrs, color=color_lr, linewidth=1.0, linestyle="--", label="LR")
    ax2.tick_params(axis="y", labelcolor=color_lr)
    ax2.legend(loc="upper right", fontsize=11)

    fig.tight_layout()
    save_path = os.path.join(output_dir, "loss_lr_combined.png")
    plt.savefig(save_path, dpi=150)
    plt.close()
    return save_path