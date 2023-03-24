import copy
import math

import torch.nn as nn
from torch.nn.functional import log_softmax

from mgz.typing import *


# import altair as alt
# import GPUtil


def _attention(query: TensorT,
               key: TensorT,
               value: TensorT,
               mask: TensorT = None,
               dropout=None) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    '''
    https://youtu.be/0PjHri8tc1c?t=727
    :param query:
    :param key:
    :param value:
    :param mask:
    :param dropout:
    :return: Tensor of shape [Batch,Time,EmbenLen]
    '''
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = scores.softmax(dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def attention(query: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
              key: FloatTensorT['B,NHeads,OutSeqLen,EmbedLen/NHeads'],
              value: FloatTensorT['B,NHeads,OutSeqLen,EmbedLen/NHeads'],
              mask: IntTensorT['B,1,OutSeqLen,OutSeqLen'] = None,
              dropout=None) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    return _attention(query, key, value, mask, dropout)


def self_attention(query: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
                   key: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
                   value: FloatTensorT[
                       'B,NHeadsNHeads,SrcSeqLen,EmbedLen/NHeads'],
                   mask: IntTensorT['B,1,SrcSeqLen,SrcSeqLen'] = None,
                   dropout=None) -> \
        Tuple[torch.Tensor, torch.Tensor]:
    mask.unsqueeze_(1)  # same masking for all nheads, so expand at dim 1
    return _attention(query, key, value, mask, dropout)


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % n_heads == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // n_heads
        self.d_v = self.d_k

        self.h = n_heads
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    # update this for typing below
    def forward(self, query: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
                key: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
                value: FloatTensorT['B,NHeads,SrcSeqLen,EmbedLen/NHeads'],
                mask: IntTensorT['B,OutSeqLen,OutSeqLen'] = None):
        "Implements Figure 2"
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = [
            lin(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.h * self.d_k)
        )
        del query
        del key
        del value
        return self.linears[-1](x)


class Embeddings(nn.Module):
    def __init__(self, vocab_len: int, d_model: int):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab_len, d_model)
        self.d_model = d_model

    def forward(self, x) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        x = self.lut(x) * math.sqrt(self.d_model)
        return x


class EncoderLayer(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"

    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x: FloatTensorT['B,SrcSeqLen,EmbedLen'], mask):
        "Follow Figure 1 (left) for connections."
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Encoder(nn.Module):
    "Core encoder is a stack of N layers"

    def __init__(self, layer: EncoderLayer, N: int):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x: FloatTensorT['B,SrcSeqLen,EmbedLen'],
                mask: FloatTensorT['B,1,SrcSeqLen']):
        "Pass the input (and mask) through each layer in turn."
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class DecoderLayer(nn.Module):
    "Decoder is made of self-attn, src-attn, and feed forward (defined below)"

    def __init__(self, size, self_attn: MultiHeadedAttention,
                 src_attn: MultiHeadedAttention, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask: IntTensorT['B,OutSeqLen'],
                tgt_mask: IntTensorT['1,OutSeqLen']):
        """
        :param x:
        :param memory:
        :param src_mask: Typically all True unless you don't want to see all of the input
        :param tgt_mask:
        :return:
        """
        "Follow Figure 1 (right) for connections."
        m = memory
        if not (src_mask == 1).all():
            raise ValueError("src_mask should be all True")
        x = self.sublayer[0](x,
                             lambda x: self.self_attn.forward(query=x, key=x,
                                                              value=x,
                                                              mask=tgt_mask))
        # src attention will give us a mapping from the source/memory to the output sequence that is out so far up until the current inference step.
        x = self.sublayer[1](x, lambda x: self.src_attn.forward(query=x, key=m,
                                                                value=m,
                                                                mask=src_mask))
        return self.sublayer[2](x, self.feed_forward)


class Decoder(nn.Module):
    "Generic N layer decoder with masking."

    def __init__(self, layer: DecoderLayer, N: int):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
                src_mask: IntTensorT['B,OutSeqLen'],
                tgt_mask: IntTensorT['1,OutSeqLen']):
        for layer in self.layers:
            x = layer(x, memory, src_mask, tgt_mask)
        return self.norm(x)


class PredictorHead(nn.Module):
    "Define standard linear + softmax generation step."

    def __init__(self, d_model: int, vocab):
        super(PredictorHead, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return log_softmax(self.proj(x), dim=-1)


class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, max_len: int, d_model: int, drop_prob: ProbT):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=drop_prob)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)
        )
        # 0::2 means u take the 0th 2 portions of the dimension
        # it's like 1/2th
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: FloatTensorT['B,SrcSeqLen,EmbedLen']) -> \
            FloatTensorT['B,SrcSeqLen,EmbedLen']:
        assert len(
            x.shape) >= 3, "Expect (batch, sequence, embedding) dimensions in `PositionalEncoding` forward"
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


class EncoderDecoder(nn.Module):
    """
    A standard Encoder-Decoder architecture. Base for this and many
    other models.
    """

    def __init__(self, encoder: Encoder, decoder: Decoder,
                 src_embed: Embeddings,
                 tgt_embed: Embeddings, generator: PredictorHead,
                 positional_encoder: PositionalEncoding):
        super(EncoderDecoder, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator
        self.positional_encoder = positional_encoder

    def forward(self, src, tgt, src_mask, tgt_mask):
        "Take in and process masked src and target sequences."
        x = self.encode(src, src_mask)
        x = self.decode(x, src_mask, tgt, tgt_mask)
        return x

    def encode(self, src: LongTensorT['B,SrcSeqLen'], src_mask):
        x = self.src_embed(src)
        x = self.positional_encoder(x)
        return self.encoder(x, src_mask)

    def decode_train(self, memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
                     src_mask: IntTensorT['B,OutSeqLen'],
                     tgt: IntTensorT['B,OutSeqLen'], tgt_mask):
        '''
        Executes for our output at a time when doing training
        :param memory:
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        '''
        x = self.tgt_embed(tgt)
        x = self.positional_encoder(x)
        return self.decoder(x, memory, src_mask, tgt_mask)

    def decode(self, memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
               src_mask: IntTensorT['B,OutSeqLen'],
               tgt: IntTensorT['B,OutSeqLen'],
               tgt_mask: IntTensorT['1,OutSeqLen']):
        '''
        Executes for our output at a time when doing inference
        :param memory:
        :param src_mask:
        :param tgt:
        :param tgt_mask:
        :return:
        '''
        x = self.tgt_embed(tgt)
        x = self.positional_encoder(x)
        return self.decoder(x, memory, src_mask, tgt_mask)


def clones(module, N: int) -> nn.ModuleList:
    "Produce N identical layers."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps: float = 1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


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


def subsequent_mask(size: SrcSeqLen):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."

    def __init__(self, d_model: int, d_ff: int, dropout: ProbT = 0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))


def make_model(
        src_vocab: int, tgt_vocab: int, N=6, d_model=512, d_ff=2048, h=8,
        dropout: ProbT = 0.1, max_seq_len: int = 1024
) -> EncoderDecoder:
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(h, d_model)
    ff = PositionwiseFeedForward(d_model, d_ff, dropout)
    position = PositionalEncoding(max_seq_len, d_model, dropout)

    model = EncoderDecoder(
        Encoder(EncoderLayer(d_model, c(attn), c(ff), dropout), N),
        Decoder(DecoderLayer(d_model, c(attn), c(attn), c(ff), dropout), N),
        Embeddings(src_vocab, d_model),
        Embeddings(tgt_vocab, d_model),
        PredictorHead(d_model, tgt_vocab), position
    )

    # This was important from their code.
    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model

    # above im not sure why dropout is passed in, i think it could be better if u just had dropout as its own class
    #     if we have good typing annotations u can make it intuitive to relate them and add blocks together
    #     as in u know what goes in and u know what comes out. u will define dropout and other classes with
    #         types that include dimensions both variable as a function of the input and constants
