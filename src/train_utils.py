"""
训练相关的结构和方法
包含: Batch, LabelSmoothing, SimpleLossCompute, 学习率调度, 训练循环,
      数据集/分词器, 贪心解码, 句子补全
"""

import os
import time
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import Dataset, DataLoader

from transformer_model import subsequent_mask


# ============================================================
# Batch 与 Mask
# ============================================================
# We stop for a quick interlude to introduce some of the tools needed
# to train a standard encoder decoder model. First we define a batch
# object that holds the src and target sentences for training, as well
# as constructing the masks.

class Batch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src, tgt=None, pad=0):
        self.src = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt, pad):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(tgt_mask.data)
        return tgt_mask


# ============================================================
# Label Smoothing (正则化)
# ============================================================
# During training, we employed label smoothing of value
# epsilon_ls=0.1. This hurts perplexity, as the model learns to be
# more unsure, but improves accuracy and BLEU score.
#
# We implement label smoothing using the KL div loss. Instead of using
# a one-hot target distribution, we create a distribution that has
# `confidence` of the correct word and the rest of the `smoothing`
# mass distributed throughout the vocabulary.
#
# Label smoothing actually starts to penalize the model if it gets
# very confident about a given choice.

class LabelSmoothing(nn.Module):
    "Implement label smoothing."

    def __init__(self, size, padding_idx, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        assert x.size(1) == self.size
        true_dist = x.data.clone()
        true_dist.fill_(self.smoothing / (self.size - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


# ============================================================
# Loss 计算
# ============================================================

class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x, y, norm):
        x = self.generator(x)
        sloss = (
            self.criterion(
                x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
            )
            / norm
        )
        return sloss.data * norm, sloss


# ============================================================
# 学习率调度 (Optimizer)
# ============================================================
# We used the Adam optimizer with beta_1=0.9, beta_2=0.98 and
# epsilon=10^-9. We varied the learning rate over the course of
# training, according to the formula:
#
#   lrate = d_model^(-0.5) * min(step_num^(-0.5), step_num * warmup_steps^(-1.5))
#
# This corresponds to increasing the learning rate linearly for the
# first warmup_steps training steps, and decreasing it thereafter
# proportionally to the inverse square root of the step number. We
# used warmup_steps=4000.
#
# Note: This part is very important. Need to train with this setup
# of the model.

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
        model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


# ============================================================
# 训练循环
# ============================================================
# Next we create a generic training and scoring function to keep track
# of loss. We pass in a generic loss compute function that also handles
# parameter updates.

def run_epoch(data_loader, model, loss_compute, optimizer, scheduler, mode="train"):
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    device = next(model.parameters()).device

    for i, batch in enumerate(data_loader):
        batch.src = batch.src.to(device)
        batch.tgt = batch.tgt.to(device)
        batch.src_mask = batch.src_mask.to(device)
        batch.tgt_mask = batch.tgt_mask.to(device)
        batch.tgt_y = batch.tgt_y.to(device)
        out = model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)
        loss, loss_node = loss_compute(out, batch.tgt_y, batch.ntokens)

        if mode == "train":
            loss_node.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens

        if i % 10 == 1 and mode == "train":
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                f"  Step {i:4d} | Loss: {loss / batch.ntokens:6.2f} "
                f"| Tokens/sec: {tokens / elapsed:7.1f} | LR: {lr:.2e}"
            )
            start = time.time()
            tokens = 0

        del loss, loss_node

    return total_loss / total_tokens


# ============================================================
# 字符级分词器
# ============================================================

PAD_TOKEN = "<pad>"
SOS_TOKEN = "<sos>"
EOS_TOKEN = "<eos>"


class CharTokenizer:
    """
    Character-level tokenizer:
    - Each character in the text is treated as one token
    - Special tokens: <pad>=0, <sos>=1, <eos>=2
    - Remaining characters are numbered 3, 4, 5, ... in sorted order
    """

    def __init__(self, text):
        special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN]
        chars = sorted(list(set(text)))
        self.vocab = special_tokens + chars
        self.char2id = {ch: i for i, ch in enumerate(self.vocab)}
        self.id2char = {i: ch for i, ch in enumerate(self.vocab)}
        self.pad_id = self.char2id[PAD_TOKEN]
        self.sos_id = self.char2id[SOS_TOKEN]
        self.eos_id = self.char2id[EOS_TOKEN]
        self.vocab_size = len(self.vocab)

    def encode(self, text):
        return [self.char2id[ch] for ch in text if ch in self.char2id]

    def decode(self, ids):
        return "".join(
            self.id2char[i] for i in ids
            if i not in (self.pad_id, self.sos_id, self.eos_id)
        )


class BPETokenizer:
    """
    BPE tokenizer (基于 tiktoken, 使用 gpt2 即 p50k_base 分词器):
    - gpt2 词表大小 50257, 比 cl100k_base 的 100256 小一半
    - 词表小 → 嵌入层参数少一半 → 过拟合风险大幅降低
    - 额外添加 3 个特殊 token: <pad>, <sos>, <eos>
    - 相比字符级分词, BPE 把常用词/子词合并为一个 token:
        "thought" → 1 token (字符级要 6 个)
        "the"     → 1 token (字符级要 3 个)
    - 序列更短, 语义更紧凑, 训练更高效
    """

    def __init__(self, text=None):
        import tiktoken
        self._enc = tiktoken.get_encoding("gpt2")
        self._base_vocab_size = self._enc.n_vocab  # 50257
        self.pad_id = self._base_vocab_size        # 50257
        self.sos_id = self._base_vocab_size + 1    # 50258
        self.eos_id = self._base_vocab_size + 2    # 50259
        self.vocab_size = self._base_vocab_size + 3  # 50260

    def encode(self, text):
        return self._enc.encode(text)

    def decode(self, ids):
        bpe_ids = [i for i in ids if i < self._base_vocab_size]
        if not bpe_ids:
            return ""
        return self._enc.decode(bpe_ids)


# ============================================================
# 数据集
# ============================================================
# We trained on the standard WMT 2014 English-German dataset
# consisting of about 4.5 million sentence pairs. Sentences were
# encoded using byte-pair encoding, which has a shared source-target
# vocabulary of about 37000 tokens.
#
# Sentence pairs were batched together by approximate sequence length.
# Each training batch contained a set of sentence pairs containing
# approximately 25000 source tokens and 25000 target tokens.
#
# For our sentence completion task, we use a simpler approach:
# split the raw text into fixed-length chunks.

class TextDataset(Dataset):
    """
    Construct training samples from raw text (sentence completion task):
    - Slide a window of seq_len across the text with given stride
    - Split each chunk into prompt (src) and continuation (tgt/tgt_y)
    - prompt_ratio controls the split point (default 0.5 = half and half)
    - The model learns: "given the prompt (encoder), generate the continuation (decoder)"
    - stride越小, 样本越多, 训练越久 (stride=1时样本数≈len(text)-seq_len)

    之前src=chunk, tgt_y=chunk是自编码器(复制任务), 推理时encoder只看到短prompt,
    和训练时看到完整序列的分布不同, 导致输出垃圾.
    现在src=前半(prompt), tgt_y=后半(continuation), 是真正的续写任务.
    """

    def __init__(self, text, tokenizer, seq_len, stride=1, prompt_ratio=0.5):
        self.tokenizer = tokenizer
        self.seq_len = seq_len
        all_ids = tokenizer.encode(text)
        self.samples = []
        prompt_len = int(seq_len * prompt_ratio)
        if prompt_len < 1:
            prompt_len = 1
        for i in range(0, len(all_ids) - seq_len, stride):
            chunk = all_ids[i : i + seq_len]
            src = chunk[:prompt_len]                    # 前半部分: prompt (送入encoder)
            cont = chunk[prompt_len:]                   # 后半部分: continuation (decoder要生成的)
            tgt = [tokenizer.sos_id] + cont             # decoder输入: <sos> + continuation
            tgt_y = cont + [tokenizer.eos_id]           # decoder目标: continuation + <eos> (让模型学会何时停止)
            self.samples.append((src, tgt, tgt_y))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        src, tgt, tgt_y = self.samples[idx]
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
            torch.tensor(tgt_y, dtype=torch.long),
        )


def collate_batch(batch, pad_id):
    "Pad multiple samples to the same length and form a Batch object."
    srcs, tgts, tgt_ys = zip(*batch)
    srcs = nn.utils.rnn.pad_sequence(srcs, batch_first=True, padding_value=pad_id)
    tgts = nn.utils.rnn.pad_sequence(tgts, batch_first=True, padding_value=pad_id)
    tgt_ys = nn.utils.rnn.pad_sequence(tgt_ys, batch_first=True, padding_value=pad_id)
    tgt_full = torch.cat([tgts, tgt_ys[:, -1:]], dim=1)
    return Batch(srcs, tgt_full, pad_id)


# ============================================================
# 贪心解码 (Greedy Decoding)
# ============================================================
# This code predicts a translation using greedy decoding for simplicity.

def greedy_decode(model, src, src_mask, max_len, start_symbol, end_symbol, temperature=1.0):
    """
    Greedy decode with optional temperature sampling.
    temperature=0: 纯贪心 (argmax), 容易陷入循环
    temperature=1: 标准采样
    temperature=0.7: 稍微保守的采样, 推荐用于推理

    注意: model.generator 返回 log_softmax, 不是原始logits.
    这里用 model.generator.proj 获取原始logits, 再做 temperature 缩放后 softmax.
    如果直接对 log_softmax 的结果做 softmax(logits/T), 等价于 p^(1/T),
    T<1 会让分布更尖锐(更贪心), 和预期相反.
    """
    import torch.nn.functional as F
    memory = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for _ in range(max_len - 1):
        out = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        raw_logits = model.generator.proj(out[:, -1])
        if temperature <= 0:
            _, next_word = torch.max(raw_logits, dim=1)
        else:
            probs = F.softmax(raw_logits / temperature, dim=-1)
            next_word = torch.multinomial(probs, 1).squeeze(-1)
        next_word = next_word.data[0]
        if next_word.item() == end_symbol:
            break
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys


def complete_sentence(model, tokenizer, prompt, max_len=200, device="cpu", temperature=0.7):
    """
    Given an English text prefix, the model auto-completes the rest.
    src = prompt字符序列 (送入encoder)
    decoder从 <sos> 开始, 逐字生成续写内容
    遇到 <eos> 或达到 max_len 停止
    返回 prompt + 续写内容 (完整句子)
    """
    model.eval()
    src_ids = tokenizer.encode(prompt)
    if len(src_ids) == 0:
        src_ids = [tokenizer.sos_id]
    src = torch.tensor([src_ids], dtype=torch.long).to(device)
    src_mask = (src != tokenizer.pad_id).unsqueeze(-2).to(device)
    result = greedy_decode(
        model, src, src_mask, max_len,
        start_symbol=tokenizer.sos_id,
        end_symbol=tokenizer.eos_id,
        temperature=temperature,
    )
    generated_ids = result[0].tolist()
    generated_ids = [i for i in generated_ids if i not in (tokenizer.pad_id, tokenizer.sos_id, tokenizer.eos_id)]
    continuation = tokenizer.decode(generated_ids)
    continuation = " ".join(continuation.split())
    return prompt + " " + continuation