# Transformer 数据流图（张量形状变化：参数名 → 实际值）

> ⚠️ **最容易看错的一点，先说清楚**：配置里的 `seq_len=64` 是"整块样本"的长度，**模型实际收到的输入是 `src=(128, 32)`、`tgt=(128, 33)`，任何形状里都不会出现 64**。
> 切分逻辑（`train_utils.py` 的 `TextDataset`）：64 个 token 的样本 → 前 32 个作为 prompt（`src`，进编码器）→ 后 32 个加 `<sos>` 作为 `tgt`（进解码器）、加 `<eos>` 作为 `tgt_y`（loss 标签）。

> 依据 `transformer_model.py` 代码整理，只关注 **形状怎么变**，不改变任何原有代码。
> 图见 [transformer_dataflow.png](transformer_dataflow.png)，可用 `python draw_dataflow.py` 重新生成。

## 参数对照表（全部来自代码，无猜测值）

| 参数名 | 值 | 出处 |
|---|---|---|
| `batch_size` | 128 | `train_runner.py` |
| **`src_len`（模型实际输入长度）** | **32** = `int(seq_len × prompt_ratio)` = int(64×0.5)，src 只取样本前半 prompt | `train_utils.py` `TextDataset` |
| **`tgt_len`（模型实际输入长度）** | **33** = 1(`<sos>`) + 32(续写部分)，由 `collate_batch` 拼接 + `Batch` 切 `[:, :-1]` 得到 | `TextDataset` + `train_utils.py` |
| `seq_len`（⚠️ 只是切块配置长度，不是模型输入形状） | 64 | `train_runner.py` |
| `d_model` | 512 | `make_model` |
| `N`（层数） | 6 | `make_model` |
| `h`（头数） | 8 | `make_model` |
| `d_k = d_model/h` | 64 | `MultiHeadedAttention` |
| `d_ff` | 2048 | `make_model` |
| `vocab_size` | 50260 = 50257(gpt2 BPE) + 3(pad/sos/eos) | `train_runner.py` `BPETokenizer` |
| `max_len`（位置编码缓存） | 5000 | `PositionalEncoding`，`pe` 形状 `(1, 5000, 512)`，按实际长度截取 |

![数据流图](transformer_dataflow.png)

---

## 0. 总览：一条链看完全程

```text
编码器:  src (batch_size, src_len) = (128, 32)
          │ Embeddings × √d_model
          ▼
        (batch_size, src_len, d_model) = (128, 32, 512)
          │ +PositionalEncoding → Encoder × 6（形状不变）
          ▼
        memory (batch_size, src_len, d_model) = (128, 32, 512)

解码器:  tgt (batch_size, tgt_len) = (128, 33)
          │ Embeddings + PositionalEncoding
          ▼
        (batch_size, tgt_len, d_model) = (128, 33, 512)
          │ Decoder × 6（形状不变，交叉注意力读 memory）
          ▼
        (batch_size, tgt_len, d_model) = (128, 33, 512)

输出:    Generator ──▶ (batch_size, tgt_len, vocab_size) = (128, 33, 50260)
```

**一句话总结**：整个 Transformer 里只有 3 个维度在变——
`d_model`（只在 FFN 中间短暂变成 `d_ff`）、`d_k/h`（只在注意力内部出现）、`vocab_size`（只在最开头是 id、最结尾是概率）。
中间所有层都是 `(B, L, d_model)` 进、`(B, L, d_model)` 出。

---

## 1. 输入与嵌入（`Embeddings` + `PositionalEncoding`）

| 步骤 | 代码 | 形状（参数名） | 实际值 |
|---|---|---|---|
| 源序列 token id | `batch.src` | `(batch_size, src_len)` | `(128, 32)`，int64 |
| 源掩码 | `(src != pad).unsqueeze(-2)` | `(batch_size, 1, src_len)` | `(128, 1, 32)`，bool |
| 嵌入查表 | `lut(x)` | `(batch_size, src_len, d_model)` | `(128, 32)` → `(128, 32, 512)` |
| 缩放 | `× math.sqrt(d_model)` | 同上（数值放大，形状不变） | `× 22.6` |
| 位置编码 | `x + pe[:, :src_len]` | `(batch_size, src_len, d_model)` | `(128, 32, 512)` |
| Dropout | `dropout(x)` | 形状不变 | — |

> `src_embed = nn.Sequential(Embeddings, PositionalEncoding)`，所以"嵌入层"一次输出就是 +PE 之后的结果。

目标侧同理，但 tgt 是 **`<sos>` + 32 个续写 token = 33**（`TextDataset` 里 `tgt = [sos_id] + cont`）：

```text
tgt          (batch_size, tgt_len)          = (128, 33)
tgt_y        (batch_size, tgt_len)          = (128, 33)     # loss 的标签，= tgt 右移一位
tgt_mask     (batch_size, tgt_len, tgt_len) = (128, 33, 33)
             # = padding掩码 (128, 1, 33) & subsequent_mask (1, 33, 33) 广播而来
```

---

## 2. 编码器（`Encoder`，N=6 层堆叠）

每层 `EncoderLayer` 内部（**入口出口都是 `(batch_size, src_len, d_model) = (128, 32, 512)`**）：

```text
x (128, 32, 512)
 │
 ├─▶ 多头自注意力 self_attn(x, x, x, src_mask) ──▶ (128, 32, 512)
 │    SublayerConnection: x + Dropout(sub(LayerNorm(x))) ──▶ (128, 32, 512)
 │
 └─▶ FFN:  Linear d_model→d_ff ──▶ (batch_size, src_len, d_ff) = (128, 32, 2048)
           ReLU + Dropout        （形状不变）
           Linear d_ff→d_model ──▶ (128, 32, 512)
    SublayerConnection ──▶ (128, 32, 512)

6 层跑完后：Encoder 最后的 norm ──▶ memory (batch_size, src_len, d_model) = (128, 32, 512)
```

---

## 3. 解码器（`Decoder`，N=6 层堆叠）

每层 `DecoderLayer` 内部（**入口出口同样是 `(batch_size, tgt_len, d_model) = (128, 33, 512)`**）：

```text
x (128, 33, 512)          memory (128, 32, 512)
 │                          │
 ├─▶ Masked多头自注意力 self_attn(x, x, x, tgt_mask) ──▶ (128, 33, 512)
 │    + Add&Norm ──▶ (128, 33, 512)
 │
 ├─▶ 交叉注意力 src_attn(Q=x, K=memory, V=memory, src_mask) ──▶ (128, 33, 512)
 │    + Add&Norm ──▶ (128, 33, 512)
 │
 └─▶ FFN: (128, 33, 512) → (128, 33, 2048) → (128, 33, 512) + Add&Norm

6 层跑完后：Decoder 最后的 norm ──▶ (batch_size, tgt_len, d_model) = (128, 33, 512)
```

---

## 4. 多头注意力内部（形状变化最多的地方）

以编码器自注意力为例（`seq_len` 即 `src_len=32`），`h=8`，`d_k=64`：

| 步骤 | 操作 | 形状（参数名） | 实际值 |
|---|---|---|---|
| 输入 q, k, v | 同一个 x | 各 `(batch_size, seq_len, d_model)` | 各 `(128, 32, 512)` |
| mask 增头维 | `mask.unsqueeze(1)` | `(batch_size, 1, seq_len, seq_len)` | `(128, 1, 32, 32)`（广播到 8 个头） |
| 线性投影 | `lin(x)` | `(batch_size, seq_len, d_model)` | `(128, 32, 512)` |
| 切多头 | `.view(batch, seq, h, d_k)` | `(batch_size, seq_len, h, d_k)` | `(128, 32, 8, 64)` |
| 换轴 | `.transpose(1, 2)` | `(batch_size, h, seq_len, d_k)` | `(128, 8, 32, 64)` ← 这是 Q、K、V |
| 打分 | `QK^T/√d_k` | `(batch_size, h, seq_len, seq_len)` | `(128, 8, 32, 32)` ← 唯一的 L×L 项 |
| 掩码+归一化 | `masked_fill` → `softmax(-1)` | 同上 | `p_attn (128, 8, 32, 32)` |
| 加权求和 | `p_attn @ V` | `(batch_size, h, seq_len, d_k)` | `(128, 8, 32, 64)` |
| 拼回头 | `.transpose(1,2).view(batch, seq, d_model)` | `(batch_size, seq_len, d_model)` | `(128, 32, 512)` |
| 输出投影 | `linears[-1]` | `(batch_size, seq_len, d_model)` | `(128, 32, 512)` |

**三种注意力的 scores（唯一的长度平方项）对照**：

| 位置 | Q 来源 | K/V 来源 | scores 形状（参数名） | 实际值 |
|---|---|---|---|---|
| 编码器自注意力 | x | x | `(batch_size, h, src_len, src_len)` | `(128, 8, 32, 32)` |
| 解码器自注意力 | x | x | `(batch_size, h, tgt_len, tgt_len)` | `(128, 8, 33, 33)` |
| 交叉注意力 | 解码器 x | memory | `(batch_size, h, tgt_len, src_len)` | `(128, 8, 33, 32)` ← 注意两维不相等 |

---

## 5. Generator 输出与 loss

| 步骤 | 操作 | 形状（参数名） | 实际值 |
|---|---|---|---|
| 解码器输出 | — | `(batch_size, tgt_len, d_model)` | `(128, 33, 512)` |
| 投影到词表 | `generator.proj` | `(batch_size, tgt_len, vocab_size)` | `(128, 33, 50260)` |
| 概率 | `log_softmax(dim=-1)` | 同上 | 同上 |
| loss 摊平 | `.view(-1, vocab_size)` vs `tgt_y.view(-1)` | `(batch_size×tgt_len, vocab_size)` | `(4224, 50260)` vs `(4224,)` |

---

## 6. 推理时（`greedy_decode`）形状怎么走

```text
src (1, src_len) ──▶ memory (batch_size=1, src_len, d_model) = (1, 32, 512)   # 编码一次，缓存住
循环解码（最多 max_len=200 步）:
  ys (1, t) ──▶ (batch_size=1, t, d_model) = (1, t, 512)
            ──▶ Generator ──▶ (batch_size=1, t, vocab_size) = (1, t, 50260)
                              取 [:, -1] 这一步，t 每轮 +1
```

> 推理时 `t` 是逐步增长的（1, 2, 3, …），`src_len` 取决于实际输入的 prompt 长度，所以这里不写死具体值。

---

## 7. 形状速查表（训练时，与图中数值一一对应）

| 张量 | 形状（参数名） | 编码器侧实际值 | 解码器侧实际值 |
|---|---|---|---|
| 输入 id | `(batch_size, src_len)` / `(batch_size, tgt_len)` | `(128, 32)` | `(128, 33)` |
| 掩码 | `(batch_size, 1, src_len)` / `(batch_size, tgt_len, tgt_len)` | `(128, 1, 32)` | `(128, 33, 33)` |
| 嵌入+PE 后 / memory | `(batch_size, ·, d_model)` | `(128, 32, 512)` | `(128, 33, 512)` |
| FFN 中间层 | `(batch_size, ·, d_ff)` | `(128, 32, 2048)` | `(128, 33, 2048)` |
| Q/K/V（切头后） | `(batch_size, h, ·, d_k)` | `(128, 8, 32, 64)` | `(128, 8, 33, 64)` |
| 注意力分数/权重 | `(batch_size, h, ·, ·)` | `(128, 8, 32, 32)` | `(128, 8, 33, 33)` |
| 交叉注意力分数 | `(batch_size, h, tgt_len, src_len)` | — | `(128, 8, 33, 32)` |
| Generator 输出 | `(batch_size, tgt_len, vocab_size)` | — | `(128, 33, 50260)` |

---

## 8. 两个会随情况变化的值（向你汇报）

1. **`vocab_size=50260` 只在用 `BPETokenizer` 时成立**（`train_runner.py` 当前用法）。如果换 `SimpleCharTokenizer`，`vocab_size = len(special_tokens + 字符集)`，会变成一个小得多的值——形状结构不变，只有最后一维变。
2. **`src_len=32 / tgt_len=33` 来自默认 `seq_len=64, prompt_ratio=0.5`**。改这两个参数，长度就跟着变（公式：`src_len = int(seq_len×prompt_ratio)`，`tgt_len = seq_len - src_len + 1`）。当前 `collate_batch` 里每条样本长度固定相同，batch 内实际无 padding；若以后换成变长句子，`pad_sequence` 会补齐到 batch 内最长。
