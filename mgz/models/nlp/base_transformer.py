import copy
import math

import torch
import torch.nn as nn
from torch.nn.functional import log_softmax

from mgz.typing import *


# import altair as alt
# import GPUtil

class TransformerContext:
    def __init__(self):
        self.in_generation = False
        self.in_train = True
        self.in_test = False
        self.past_keys: FloatTensorT[
            'B,NHeads,OutSeqStep,EmbedLen/NHeads'] = None
        self.past_values: FloatTensorT[
            'B,NHeads,OutSeqStep,EmbedLen/NHeads'] = None

    def add_key(self,
                new_key: FloatTensorT['B,NHeads,OutSeqStep,EmbedLen/NHeads']):
        if self.past_keys is not None:
            self.past_keys = torch.cat([self.past_keys, new_key], dim=2)

    def add_value(self,
                  new_val: FloatTensorT['B,NHeads,OutSeqStep,EmbedLen/NHeads']):
        if self.past_values is not None:
            self.past_values = torch.cat([self.past_values, new_val],
                                         dim=2)

    def reset(self):
        del self


class BaseTransformer(nn.Module):

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
