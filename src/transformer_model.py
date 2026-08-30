"""
Transformer 模型架构
按论文 "Attention Is All You Need" 架构图顺序组织:
  总模型架构 → 嵌入层 → 位置编码 → 注意力 → 多头注意力 → 标准化与残差连接 → FFN → 编码器 → 解码器 → Generator & make_model
"""

import math
import copy
import torch
import torch.nn as nn
from torch.nn.functional import log_softmax


# 复制模块N次且包装在ModuleList中，使其在注册在pytorch中时自动注册参数
def clones(module, N):
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


# # 第1步：torch.ones → 全1矩阵
# [[1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1],
#  [1, 1, 1, 1]]
#
# # 第2步：torch.triu(..., diagonal=1) → 取上三角（不含对角线）
# [[0, 1, 1, 1],
#  [0, 0, 1, 1],
#  [0, 0, 0, 1],
#  [0, 0, 0, 0]]
#
# # 第3步：subsequent_mask == 0 → 取反，上三角变False
# [[True,  False, False, False],
#  [True,  True,  False, False],
#  [True,  True,  True,  False],
#  [True,  True,  True,  True ]]
def subsequent_mask(size):
    "Mask out subsequent positions."
    # 1. 定义掩码的形状: (1, size, size)
    # batch维度为1，后续可通过广播机制应用到整个batch
    attn_shape = (1, size, size)
    # 2. 生成上三角矩阵（diagonal=1 表示主对角线以上，不含对角线）
    # torch.ones(attn_shape) 生成全1矩阵
    # torch.triu(..., diagonal=1) 保留上三角部分，下三角（含对角线）置0
    # .type(torch.uint8) 转为uint8类型（旧版PyTorch写法）
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    # 返回一个布尔矩阵，True表示可以关注，False表示不能关注
    return subsequent_mask == 0


# ============================================================
# 总模型架构
# ============================================================
# Most competitive neural sequence transduction models have an
# encoder-decoder structure. Here, the encoder maps an input sequence
# of symbol representations (x_1, ..., x_n) to a sequence of
# continuous representations z = (z_1, ..., z_n). Given z, the decoder
# then generates an output sequence (y_1,..., y_m) of symbols one
# element at a time. At each step the model is auto-regressive,
# consuming the previously generated symbols as additional input
# when generating the next.
#
# The Transformer follows this overall architecture using stacked
# self-attention and point-wise, fully connected layers for both the
# encoder and decoder, shown in the left and right halves of Figure 1,
# respectively.

class EncoderDecoder(nn.Module):
    """
    encoder:编码器
    decoder:解码器
    src_embed:输入嵌入层
    tgt_embed:输出嵌入层
    generator:生成器
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def forward(self, src, tgt, src_mask, tgt_mask):
        '''
        src:输入序列
        tgt:输出序列
        src_mask:输入序列掩码
        tgt_mask:输出序列掩码
        return:输出序列
        '''
        return self.decode(self.encode(src, src_mask), src_mask, tgt, tgt_mask)

    def encode(self, src, src_mask):
        return self.encoder(self.src_embed(src), src_mask)

    def decode(self, memory, src_mask, tgt, tgt_mask):
        return self.decoder(self.tgt_embed(tgt), memory, src_mask, tgt_mask)


# ============================================================
# 嵌入层
# ============================================================
# Similarly to other sequence transduction models, we use learned
# embeddings to convert the input tokens and output tokens to vectors
# of dimension d_model. We also use the usual learned linear
# transformation and softmax function to convert the decoder output to
# predicted next-token probabilities. In our model, we share the same
# weight matrix between the two embedding layers and the pre-softmax
# linear transformation. In the embedding layers, we multiply those
# weights by sqrt(d_model).

# id->vector
# 输入：(batch_size, seq_len)
# 输出：(batch, seq_len, d_model)
class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    # Embedding 初始化后数值较小，而位置编码值域在 [-1, 1]，直接相加会导致语义信息被位置信号淹没。
    # 所以，我们对Embedding进行缩放，使Embedding的数值在 [-1, 1] 范围内。
    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# ============================================================
# 位置编码
# ============================================================
# Since our model contains no recurrence and no convolution, in order
# for the model to make use of the order of the sequence, we must
# inject some information about the relative or absolute position of
# the tokens in the sequence. To this end, we add "positional
# encodings" to the input embeddings at the bottoms of the encoder
# and decoder stacks. The positional encodings have the same dimension
# d_model as the embeddings, so that the two can be summed.
#
# In this work, we use sine and cosine functions of different frequencies:
#
#   PE_(pos,2i)   = sin(pos / 10000^(2i/d_model))
#   PE_(pos,2i+1) = cos(pos / 10000^(2i/d_model))
#
# where pos is the position and i is the dimension. That is, each
# dimension of the positional encoding corresponds to a sinusoid. The
# wavelengths form a geometric progression from 2*pi to 10000 * 2*pi.
# We chose this function because we hypothesized it would allow the
# model to easily learn to attend by relative positions, since for any
# fixed offset k, PE_(pos+k) can be represented as a linear function
# of PE_pos.
#
# In addition, we apply dropout to the sums of the embeddings and the
# positional encodings in both the encoder and decoder stacks. For the
# base model, we use a rate of P_drop=0.1.
#
# We also experimented with using learned positional embeddings instead,
# and found that the two versions produced nearly identical results.
# We chose the sinusoidal version because it may allow the model to
# extrapolate to sequence lengths longer than the ones encountered
# during training.

class PositionalEncoding(nn.Module):
    "Implement the PE function."
    '''
        d_model: 输入维度
        dropout: dropout概率
        max_len: 最大序列长度
    '''

    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)  # (max_len, d_model) 全零矩阵
        # 先生成长为max_len的tensor,然后添加一个维度,变成(max_len, 1)的tensor
        position = torch.arange(0, max_len).unsqueeze(1)  # (max_len, 1) 位置索引
        # torch.arange(0, d_model, 2)=2i（1）
        # -ln(10000)/d_model（2）
        # （1）×（2）=-2i*ln(10000)/d_model
        # torch.exp(（1）*（2）)=10000^(2i/d_model)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 所有行、从第0列开始、步长为2的列 → 偶数维度，赋值sin
        pe[:, 0::2] = torch.sin(position * div_term)
        # 所有行、从第1列开始、步长为2的列 → 奇数维度，赋值cos
        pe[:, 1::2] = torch.cos(position * div_term)
        # 增加batch维度: (1, max_len, d_model)，方便后续广播
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        # 按实际序列长度截取位置编码（因为 pe 是按最大长度预计算的，实际 seq_len 可能更短）。
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)  # x+pe
        return self.dropout(x)


# ============================================================
# 注意力 (Scaled Dot-Product Attention)
# ============================================================
# An attention function can be described as mapping a query and a set
# of key-value pairs to an output, where the query, keys, values, and
# output are all vectors. The output is computed as a weighted sum of
# the values, where the weight assigned to each value is computed by a
# compatibility function of the query with the corresponding key.
#
# We call our particular attention "Scaled Dot-Product Attention".
# The input consists of queries and keys of dimension d_k, and values
# of dimension d_v. We compute the dot products of the query with all
# keys, divide each by sqrt(d_k), and apply a softmax function to
# obtain the weights on the values.
#
# In practice, we compute the attention function on a set of queries
# simultaneously, packed together into a matrix Q. The keys and values
# are also packed together into matrices K and V. We compute the
# matrix of outputs as:
#
#   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V

def attention(query, key, value, mask=None, dropout=None):
    "计算注意力权重"
    d_k = query.size(-1)  # 查询向量的维度
    # 先将key的最后两维度转置，计算QK^T的矩阵，再除以sqrt{d_k}
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # 将 mask 为 0 的位置填充为 -1e9，确保在softmax中被忽略掉
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    # softmax函数，将每个位置的注意力权重归一化到0-1之间，且每个位置的权重和为1
    p_attn = scores.softmax(dim=-1)
    # 应用dropout
    if dropout is not None:
        p_attn = dropout(p_attn)
    # 最后一步返回softmax(QK^T/sqrt{d_k})V和注意力权重矩阵(softmax(QK^T/sqrt{d_k}))
    return torch.matmul(p_attn, value), p_attn


# ============================================================
# 多头注意力
# ============================================================

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        '''
        h: 多头头数
        d_model: 模型维度
        dropout: dropout概率
        '''
        super(MultiHeadedAttention, self).__init__()
        # 确保模型维度可以被头数整除
        assert d_model % h == 0
        # 计算每个头的维度,每个头的维度相同
        self.d_k = d_model // h
        self.h = h
        # 4个线性层，用于将输入映射到每个头的维度,创建 4 个相同的线性层：前 3 个分别投影 Q/K/V，第 4 个是输出投影 W^O
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        # 注意力机制
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        "Implements Figure 2"
        # mask是掩码矩阵，用于在计算注意力权重时忽略一些位置
        # 一般mask的形状为(batch, seq_len, seq_len)
        # 而scores的形状为(batch, h, seq_len, seq_len)
        # 为了将mask应用到每个头，需要在维度 1 上添加一个维度
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)  # query的第一个维度，即批量大小

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # zip将括号里面的封装成元组
        # 这里self.linears就是元组a
        # query, key, value就是元组b
        # 有三个循环
        # 第一次循环lin=self.linears[0],x=query
        # 第二次循环lin=self.linears[1],x=key
        # 第三次循环lin=self.linears[2],x=value
        # 第一次循环
        # lin=self.linears[0],x=query
        # lin(x)=self.linears[0](query)，也就是把x作为输入，通过线性层映射到每个头的维度
        # 假设x=query: (batch, seq_len, d_model),lin(x)
        # view的功能是将(batch, seq_len, d_model)转换为(batch, seq_len, h, d_k)),d_model=h×d_k
        # -1就是自动计算维度，这里自动计算出seq_len
        # transpose(1, 2)的功能是将第1个和第2个维度交换
        # 将(batch, seq_len, h, d_k))转换为(batch, h, seq_len, d_k)
        # 最后得到了Q,K,V这3个矩阵
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        # attention函数计算注意力权重,返回注意力权重矩阵(softmax(QK^T/sqrt{d_k}))和注意力输出(softmax(QK^T/sqrt{d_k})V)
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        # transpose 后（逻辑 vs 内存不一致）：
        # 逻辑：(3, 2) → [[1,4], [2,5], [3,6]]
        # 内存：[1, 2, 3, 4, 5, 6]  ← 跳着读才能得到逻辑结果
        #
        # contiguous() 后（逻辑 vs 内存一致）：
        # 逻辑：(3, 2) → [[1,4], [2,5], [3,6]]
        # 内存：[1, 4, 2, 5, 3, 6]  ← 按逻辑顺序重新排列，紧挨着读
        # (batch, h, seq_len, d_k)交换下标1和2，得到(batch, seq_len, h, d_k)
        # contiguous()函数将内存布局转换为连续的，即将内存布局转换为(batch, seq_len, h, d_k)
        # view将(batch, seq_len, h, d_k)转换为(batch, seq_len, h * d_k)，
        # 也就是将每个头的输出拼接起来，得到(batch, seq_len, h * d_k)
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        # 回收
        del query
        del key
        del value
        # 合并线性层输出
        return self.linears[-1](x)


# ============================================================
# 标准化 (LayerNorm) 与 残差连接 (SublayerConnection)
# ============================================================
# We employ a residual connection around each of the two sub-layers,
# followed by layer normalization.
#
# That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)),
# where Sublayer(x) is the function implemented by the sub-layer
# itself. We apply dropout to the output of each sub-layer, before it
# is added to the sub-layer input and normalized.
#
# To facilitate these residual connections, all sub-layers in the
# model, as well as the embedding layers, produce outputs of dimension
# d_model=512.

class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # 缩放因子γ
        self.b_2 = nn.Parameter(torch.zeros(features))  # 偏置项β
        self.eps = eps  # 小常量，防止除0错误

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)  # 按行计算均值，保留维度，方便后续广播
        std = x.std(-1, keepdim=True)  # 按行计算标准差，保留维度，方便后续广播
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


# 先层归一化，再应用子层，最后添加残差连接，再应用dropout层
# 此代码: x + Dropout(Sublayer(Norm(x)))
# 这与原文中的实现不同，原文中的实现是先应用子层和残差连接，再层归一化
# 原文: SublayerOutput(x) = LayerNorm(x + Dropout(Sublayer(x)))
# 先归一化不用warmup，先让其稳定就不用warmup了
class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


# ============================================================
# 前馈神经网络 (Position-wise Feed-Forward)
# ============================================================
#
#   FFN(x) = max(0, xW_1 + b_1) W_2 + b_2
#


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    '''
        d_model: 输入维度
        d_ff: 中间层维度
        dropout: dropout概率
    '''

    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)  # 扩展到高维度，d_ff维度
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


# ============================================================
# 编码器 (Encoder)
# ============================================================

class EncoderLayer(nn.Module):
    "自注意力层+前馈层"
    '''
    size: d_model
    self_attn: 自注意力层
    feed_forward: 前馈层
    dropout: dropout率
    '''

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        # sublayer[0]：包含自注意力层的残差连接
        # sublayer[1]：包含前馈层的残差连接
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        "Follow Figure 1 (left) for connections."
        # sublayer[0]：包含自注意力层的残差连接
        # sublayer[0]((self, x, sublayer)),返回：x + self.dropout(sublayer(self.norm(x))),
        # sublayer(self.norm(x))就是lambda x: self.self_attn(norm(x), norm(x), norm(x), mask)
        # mask：屏蔽mask位置的元素，并非因果掩码，而是屏蔽<pad>等特殊符号
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))#传入两个参数
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)  # 拷贝n个EncoderLayer模块
        self.norm = LayerNorm(layer.size)  # 层归一化

    # 这个mask就是src_mask，用于屏蔽padding
    def forward(self, x, mask):
        # 屏蔽mask位置的元素，并非因果掩码，而是屏蔽<pad>等特殊符号
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


# ============================================================
# 解码器 (Decoder)
# ============================================================

class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask):
        "Follow Figure 1 (right) for connections."
        m = memory  # encoder输出,就是那个output输入到decoder
        # sublayer[0]：包含自注意力层的残差连接
        # sublayer[0]((self, x, sublayer)),返回：x + self.dropout(sublayer(self.norm(x))),
        # sublayer(self.norm(x))就是lambda x: self.self_attn(norm(x), norm(x), norm(x), tgt_mask)
        # 这个是带因果掩码的自注意力层，用于屏蔽未来位置
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        # Cross-Attention:   Q=x,  K=m,  V=m    → 用 Encoder 的信息查询
        # 这个是带padding掩码的跨注意力层，用于屏蔽padding
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        # 拷贝n个DecoderLayer模块
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    '''
        x: decoder自己的输入
        memory: encoder输出
        src_mask: 源序列的掩码，用于屏蔽padding
        tgt_mask: 目标序列的掩码，因果+padding掩码
    '''
    def forward(self, x, memory, src_mask, tgt_mask):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


# ============================================================
# Generator & make_model (编码器-解码器最终组装)
# ============================================================
# Here we define a function from hyperparameters to a full model.
# This was important from their code: Initialize parameters with
# Glorot / fan_avg.

# 生成器
class Generator(nn.Module):
    '''
    d_model: 模型维度
    vocab: 词汇表大小
    proj: 投影层
    '''

    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


'''
    src_vocab: 源语言词表大小，决定 Encoder 嵌入层和输入维度
    tgt_vocab: 目标语言词表大小，决定 Decoder 嵌入层及 Generator 输出维度
    N: Encoder/Decoder 堆叠层数，越大模型越深、容量越强，但计算量线性增长
    d_model: 模型内部统一隐藏维度，所有子层（注意力、FFN、嵌入）均在此维度上运算
    d_ff: 前馈网络维度
    h: 多头注意力头数
    dropout: dropout概率
'''
def make_model(
    src_vocab, tgt_vocab, N=6, d_model=512, d_ff=2048, h=8, dropout=0.1
):
    "Helper: Construct a model from hyperparameters."
    # 克隆子层，用于堆叠 N 层
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(d_model, dropout)
    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        nn.Sequential(Embeddings(d_model, src_vocab), c(position)),
        nn.Sequential(Embeddings(d_model, tgt_vocab), c(position)),
        Generator(d_model, tgt_vocab),
    )

    # 权重共享 (Weight Tying, 论文 Section 3):
    # "We share the same weight matrix between the two embedding layers
    #  and the pre-softmax linear transformation"
    # 即: src_embed.weight = tgt_embed.weight = generator.proj.weight
    # 好处: 大幅减少参数量 (嵌入层只算一份), 且论文表明能提升效果
    if src_vocab == tgt_vocab:
        model.src_embed[0].lut.weight = model.tgt_embed[0].lut.weight
        model.generator.proj.weight = model.tgt_embed[0].lut.weight

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    # 对requires_grad=True以及维度大于1的参数进行初始化，使用Glorot / fan_avg初始化
    # 因为维度为1的参数（如偏置项）不需要初始化，所以这里只初始化维度大于1的参数
    for p in model.parameters():
        if p.dim() > 1:
            # Xavier Uniform 初始化：权重 ~ U(-a, a)，其中 a = gain * sqrt(6 / (fan_in + fan_out))
            # 目标：保持前向/反向传播中信号方差稳定，避免梯度消失或爆炸
            nn.init.xavier_uniform_(p)
    return model