from __future__ import annotations

from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch
from mgz.models.nlp.bert_basic import subsequent_mask, EncoderDecoder
from mgz.typing import *


def train_forward(model: EncoderDecoder, batch: SentenceBatch):
    return model.forward(batch.src, batch.tgt, batch.src_mask, batch.tgt_mask)


def greedy_decode(model: EncoderDecoder, src: IntTensorT['B,SrcSeqLen'],
                  src_mask: IntTensorT['B,1,SrcSeqLen'], max_len: int,
                  start_symbol: int):
    memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = model.encode(src, src_mask)
    ys = torch.zeros(1, 1).fill_(start_symbol).type_as(src.data)
    for i in range(max_len - 1):
        print('subsequent mask ', subsequent_mask(ys.size(1)).shape)
        out: FloatTensorT['B,ys.size(1),EmbedLen'] = model.decode(
            memory, src_mask, ys, subsequent_mask(ys.size(1)).type_as(src.data)
        )
        # THIS IS NOT THE RIGHT WAY TO DO THIS, ACTUALLY YOU SHOULD BE USING THE LAST TOKEN, AND MASKING PREVIOUS ONES IN THE SELF ATTENTION. SO YOUR QUERY VECTOR IS ONLY 1 IN SEQUENCE DIMENSION, WHILE KEY AND VALUE ARE GONNA GROW EVERY DECODING STEP. THAT WAY
        last_token_pred: FloatTensorT['B,EmbedLen'] = out[:, -1]
        prob = model.generator(last_token_pred)
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.zeros(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )
    return ys
