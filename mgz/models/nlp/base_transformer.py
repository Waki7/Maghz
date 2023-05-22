import copy
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from mgz.typing import *


# import altair as alt
# import GPUtil

class TransformerContext:
    def __init__(self, b: int, embed_len: int):
        self.in_generation = False
        self.in_train = True
        self.in_test = False
        self.past_keys: FloatTensorT[
            'B,OutSeqStep,EmbedLen'] = torch.ones((b, 0, embed_len))
        self.past_values: FloatTensorT[
            'B,OutSeqStep,EmbedLen'] = torch.ones((b, 0, embed_len))

    def add_key(self,
                new_key: FloatTensorT['B,OutSeqStep,EmbedLen']):
        self.past_keys = torch.cat([self.past_keys, new_key], dim=2)

    def add_value(self,
                  new_val: FloatTensorT['B,OutSeqStep,EmbedLen']):
        self.past_values = torch.cat([self.past_values, new_val],
                                     dim=2)

    def reset(self):
        del self


class BaseTransformer(nn.Module):
    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        self.config = config

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def decode(self,
               encoder_memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
               tgt_ids: LongTensorT['B,OutSeqLen'],
               src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen'],
               tgt_mask: IntTensorT['B,OutSeqLen,OutSeqLen'],
               transformer_ctx: TransformerContext):
        raise NotImplementedError

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,OutSeqLen'],
            src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,OutSeqLen,OutSeqLen']
    ) -> FloatTensorT['B,OutSeqLen,EmbedLen']:
        raise NotImplementedError

    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,OutSeqLen'],
                 src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen']):
        context = TransformerContext(src_ids.shape[0], self.config.d_model)
        context.in_generation = True
        memory: FloatTensorT['B,OutSeqLen,EmbedLen'] = self.encode(
            src_ids=src_ids, src_mask=src_mask)
        last_token = tgt_ids
        for i in range(0, 12):
            decoder_hidden_state: FloatTensorT['B,1,EmbedLen'] = \
                self.decode(encoder_memory=memory,
                            tgt_ids=last_token,
                            src_mask=src_mask,
                            transformer_ctx=context)
            last_token = torch.softmax(self.lm_head(decoder_hidden_state),
                                       dim=-1).argmax(dim=-1)
            tgt_ids = torch.cat([tgt_ids, last_token], dim=-1)
        return tgt_ids
