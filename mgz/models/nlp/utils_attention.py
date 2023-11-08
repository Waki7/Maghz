import torch.utils.checkpoint
from torch import nn

import settings
from mgz.typing import *


def _led_attention():
    pass


def _attention(query: FloatTensorT,
               key: FloatTensorT,
               value: FloatTensorT,
               mask: IntTensorT = None,
               dropout_p: float = None) -> \
        Tuple[FloatTensorT['B,SrcSeqLen,EmbedLen'], torch.Tensor]:
    "Compute 'Scaled Dot Product Attention'"
    scores: FloatTensorT['B,TgtSeqLen,SrcSeqLen'] = \
        torch.matmul(query, key.transpose(-2, -1))
    # same mask per head
    if mask is not None:
        assert mask.shape[0] == scores.shape[
            0], 'make sure mask aligns score shape {} vs mask shape {} '.format(
            scores.shape, mask.shape)
        mask = mask.unsqueeze(1)
        scores = scores.masked_fill(mask == 0, -1e4)

    # also called attention weights
    p_attn: torch.Tensor = scores.softmax(dim=-1)
    # if not in training, None will be passed
    if dropout_p is not None:
        p_attn = nn.functional.dropout(p_attn, p=dropout_p)

    # using the probabilities to pay attention to the value
    # noinspection PyTypeChecker
    attn_out = torch.matmul(p_attn, value)
    del mask
    del scores
    del p_attn
    settings.empty_cache()
    return attn_out


def attention(query: FloatTensorT['B*NHeads,TgtSeqLen,EmbedLen/NHeads'],
              key: FloatTensorT['B*NHeads,SrcSeqLen,EmbedLen/NHeads'],
              value: FloatTensorT['B*NHeads,SrcSeqLen,EmbedLen/NHeads'],
              mask: IntTensorT['B*NHeads,TgtSeqLen,SrcSeqLen'] = None,
              dropout_p: ProbT = None) -> \
        Tuple[FloatTensorT['B,TgtSeqLen,EmbedLen'], torch.Tensor]:
    return _attention(query, key, value, mask, dropout_p)


def self_attention(query: FloatTensorT['B*NHeads,SrcSeqLen,EmbedLen/NHeads'],
                   key: FloatTensorT['B*NHeads,SrcSeqLen,EmbedLen/NHeads'],
                   value: FloatTensorT[
                       'B*NHeads,SrcSeqLen,EmbedLen/NHeads'],
                   mask: IntTensorT['B*NHeads,SrcSeqLen,SrcSeqLen'] = None,
                   dropout_p=None) -> \
        Tuple[FloatTensorT['B,SrcSeqLen,EmbedLen'], torch.Tensor]:
    return _attention(query, key, value, mask, dropout_p)
