# -*- coding: utf-8 -*-
"""
生成 Transformer 数据流图（张量形状变化）
每个形状都用「参数名 + 具体数值」对照标注。
依据 transformer_model.py + train_runner.py/train_utils.py 的实际配置绘制。
输出 transformer_dataflow.png
运行: python draw_dataflow.py
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei"]
plt.rcParams["axes.unicode_minus"] = False

# 配色
C_ENC = "#dbeafe"; C_ENC_E = "#3b82f6"    # 编码器 蓝
C_DEC = "#ffedd5"; C_DEC_E = "#f97316"    # 解码器 橙
C_GEN = "#dcfce7"; C_GEN_E = "#22c55e"    # 生成器 绿
C_MASK = "#fef9c3"; C_MASK_E = "#ca8a04"  # 掩码 黄
C_MHA = "#ede9fe"; C_MHA_E = "#8b5cf6"    # 多头注意力 紫
C_IN = "#f1f5f9"; C_IN_E = "#64748b"      # 输入 灰

fig, ax = plt.subplots(figsize=(14.5, 13))
ax.set_xlim(0, 130)
ax.set_ylim(0, 120)
ax.axis("off")


def box(cx, cy, w, h, lines, fc, ec, fs=9, dashed=False, colors=None):
    """画一个圆角框；lines 每行一条文字；colors 可指定每行颜色（默认第一行加粗）"""
    p = FancyBboxPatch(
        (cx - w / 2, cy - h / 2), w, h,
        boxstyle="round,pad=0.3,rounding_size=1.2",
        facecolor=fc, edgecolor=ec, linewidth=1.4,
        linestyle="--" if dashed else "-",
    )
    ax.add_patch(p)
    n = len(lines)
    for i, t in enumerate(lines):
        y = cy + h / 2 - (i + 0.5) * (h / n)
        c = colors[i] if colors else ("#111827" if i == 0 else "#374151")
        weight = "bold" if i == 0 else "normal"
        ax.text(cx, y, t, ha="center", va="center", fontsize=fs, weight=weight, color=c)


def arrow(x1, y1, x2, y2, color="#334155", dashed=False, lw=1.6):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle="-|>", mutation_scale=14,
        color=color, linewidth=lw,
        linestyle="--" if dashed else "-",
        shrinkA=1, shrinkB=1,
    )
    ax.add_patch(a)


def shape2(lines):
    """三行盒子：模块名 / 参数名形状 / 具体数值"""
    return lines


# ============ 标题 ============
ax.text(65, 117, "Transformer 数据流图（张量形状变化：参数名 → 实际值）",
        ha="center", va="center", fontsize=15, weight="bold")
ax.text(65, 113.6,
        "参数取值（train_runner.py 实际配置）：batch_size=128, seq_len=64, prompt_ratio=0.5 → src_len=32(prompt), tgt_len=33(<sos>+续写32) | "
        "d_model=512, N=6, h=8, d_k=64, d_ff=2048 | vocab_size=50260 (gpt2 BPE 50257+3)",
        ha="center", va="center", fontsize=8, color="#475569")

# ============ 编码器列 ============
EX, EW = 26, 34
box(EX, 106.25, EW, 7.5, shape2(["输入 src（token id）",
                                 "(batch_size, src_len)",
                                 "= (128, 32)"]), C_IN, C_IN_E)
box(EX, 96, EW, 8, shape2(["Embeddings × √d_model",
                           "(batch_size, src_len, d_model)",
                           "= (128, 32, 512)"]), C_ENC, C_ENC_E)
box(EX, 85.5, EW, 8, shape2(["PositionalEncoding + Dropout",
                             "(batch_size, src_len, d_model)",
                             "= (128, 32, 512)"]), C_ENC, C_ENC_E)
box(EX, 73, EW, 12, shape2(["EncoderLayer × 6（N=6）",
                            "多头自注意力(Q=K=V, mask=src_mask) + Add&Norm",
                            "FFN：d_model → d_ff → d_model + Add&Norm",
                            "形状不变：(batch_size, src_len, d_model) = (128, 32, 512)"]),
    C_ENC, C_ENC_E, fs=8)
box(EX, 60.5, EW, 8, shape2(["LayerNorm → 输出 memory",
                             "(batch_size, src_len, d_model)",
                             "= (128, 32, 512)"]), C_ENC, C_ENC_E)

arrow(EX, 102.5, EX, 100)
arrow(EX, 92, EX, 89.5)
arrow(EX, 81.5, EX, 79)
arrow(EX, 67, EX, 64.5)
ax.text(5, 101, "加了一维 d_model\n2D → 3D", ha="center", va="center",
        fontsize=7, color="#64748b")

# ============ 解码器列 ============
DX, DW = 99, 34
box(DX, 106.25, DW, 7.5, shape2(["输出 tgt（<sos>+续写，右移一位）",
                                 "(batch_size, tgt_len)",
                                 "= (128, 33)"]), C_IN, C_IN_E)
box(DX, 96, DW, 8, shape2(["Embeddings × √d_model",
                           "(batch_size, tgt_len, d_model)",
                           "= (128, 33, 512)"]), C_DEC, C_DEC_E)
box(DX, 85.5, DW, 8, shape2(["PositionalEncoding + Dropout",
                             "(batch_size, tgt_len, d_model)",
                             "= (128, 33, 512)"]), C_DEC, C_DEC_E)
box(DX, 72, DW, 14, shape2(["DecoderLayer × 6（N=6）",
                            "Masked自注意力(mask=tgt_mask) + Add&Norm",
                            "交叉注意力(Q=x, K=V=memory, mask=src_mask) + Add&Norm",
                            "FFN：d_model → d_ff → d_model + Add&Norm",
                            "形状不变：(batch_size, tgt_len, d_model) = (128, 33, 512)"]),
    C_DEC, C_DEC_E, fs=7.5)
box(DX, 58.5, DW, 8, shape2(["LayerNorm",
                             "(batch_size, tgt_len, d_model)",
                             "= (128, 33, 512)"]), C_DEC, C_DEC_E)
box(DX, 47.5, DW, 8.5, shape2(["Generator：Linear + log_softmax",
                               "(batch_size, tgt_len, vocab_size)",
                               "= (128, 33, 50260)"]), C_GEN, C_GEN_E)

arrow(DX, 102.5, DX, 100)
arrow(DX, 92, DX, 89.5)
arrow(DX, 81.5, DX, 79)
arrow(DX, 65, DX, 62.5)
arrow(DX, 54.5, DX, 51.75)

# ============ 掩码（虚线框，广播使用） ============
box(62.5, 98.5, 30, 8, shape2(["tgt_mask（padding + 因果）",
                               "(batch_size, tgt_len, tgt_len)",
                               "= (128, 33, 33)"]), C_MASK, C_MASK_E, fs=8, dashed=True)
box(62.5, 88.5, 30, 8, shape2(["src_mask（padding）",
                               "(batch_size, 1, src_len)",
                               "= (128, 1, 32)"]), C_MASK, C_MASK_E, fs=8, dashed=True)

# ============ memory 交叉箭头 ============
arrow(EX + EW / 2, 60.5, 62.5, 60.5)
arrow(62.5, 60.5, 62.5, 70.5)
arrow(62.5, 70.5, DX - DW / 2, 70.5)
ax.text(52, 55.2, "memory (batch_size, src_len, d_model)\n= (128, 32, 512)",
        ha="center", va="center", fontsize=7, color="#1d4ed8", weight="bold")

# ============ 多头注意力内部形状 ============
ax.text(65, 39.5, "多头注意力内部：形状变化（h=8, d_k = d_model/h = 64；图中按编码器情形，seq_len=src_len=32）",
        ha="center", va="center", fontsize=10.5, weight="bold", color="#6d28d9")

MW, MH, r1y, r2y = 28, 8, 31.5, 21.5
box(17,  r1y, MW, MH, shape2(["输入 q, k, v",
                              "(batch_size, seq_len, d_model)",
                              "= (128, 32, 512)"]), C_MHA, C_MHA_E, fs=8)
box(49,  r1y, MW, MH, shape2(["Linear + view + transpose",
                              "(batch_size, h, seq_len, d_k)",
                              "= (128, 8, 32, 64)"]), C_MHA, C_MHA_E, fs=8)
box(81,  r1y, MW, MH, shape2(["scores = QK^T/√d_k + mask",
                              "(batch_size, h, seq_len, seq_len)",
                              "= (128, 8, 32, 32)"]), C_MHA, C_MHA_E, fs=8)
box(113, r1y, MW, MH, shape2(["softmax → 注意力权重",
                              "(batch_size, h, seq_len, seq_len)",
                              "= (128, 8, 32, 32)"]), C_MHA, C_MHA_E, fs=8)
box(113, r2y, MW, MH, shape2(["权重 × V",
                              "(batch_size, h, seq_len, d_k)",
                              "= (128, 8, 32, 64)"]), C_MHA, C_MHA_E, fs=8)
box(81,  r2y, MW, MH, shape2(["transpose + view 拼接",
                              "(batch_size, seq_len, d_model)",
                              "= (128, 32, 512)"]), C_MHA, C_MHA_E, fs=8)
box(49,  r2y, MW, MH, shape2(["输出 Linear",
                              "(batch_size, seq_len, d_model)",
                              "= (128, 32, 512)"]), C_MHA, C_MHA_E, fs=8)

arrow(31, r1y, 35, r1y)
arrow(63, r1y, 67, r1y)
arrow(95, r1y, 99, r1y)
arrow(113, r1y - MH / 2, 113, r2y + MH / 2)
arrow(99, r2y, 95, r2y)
arrow(67, r2y, 63, r2y)

# ============ 三种注意力的 scores 形状 ============
ax.text(65, 13.8, "三种注意力的 scores（QK^T/√d_k）形状：",
        ha="center", va="center", fontsize=9.5, weight="bold", color="#6d28d9")
SW, SH, sy = 39, 7.5, 7.5
box(21.5, sy, SW, SH, shape2(["编码器自注意力",
                              "(batch_size, h, src_len, src_len) = (128, 8, 32, 32)"]),
    C_MHA, C_MHA_E, fs=7.5)
box(65, sy, SW, SH, shape2(["解码器自注意力",
                            "(batch_size, h, tgt_len, tgt_len) = (128, 8, 33, 33)"]),
    C_MHA, C_MHA_E, fs=7.5)
box(108.5, sy, SW, SH, shape2(["交叉注意力(Q=tgt, K/V=memory)",
                               "(batch_size, h, tgt_len, src_len) = (128, 8, 33, 32)"]),
    C_MHA, C_MHA_E, fs=7.5)

# ============ 底部注释 ============
ax.text(65, 2.2,
        "注1：tgt_len=33 是因为 decoder 输入 = <sos> + 续写部分(32)，由 collate_batch 拼接后 Batch 里再切 [:, :-1] 得到。",
        ha="center", va="center", fontsize=8, color="#475569")
ax.text(65, 0.2,
        "注2：mask 在注意力内部 unsqueeze(1) 后广播到 h 个头；除嵌入、FFN 中间层(d_ff)和 Generator(vocab_size) 外，其余层形状不变。",
        ha="center", va="center", fontsize=8, color="#475569")

# 保存到项目根的 docs/ 目录 (tools/ 的上一级)
import os
_out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "docs", "transformer_dataflow.png")
fig.savefig(_out_path, dpi=170, bbox_inches="tight",
            facecolor="white")
print(f"已生成 {_out_path}")
