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
        self.generation = False
        self.train = True
        self.test = False
        self.past_keys: FloatTensorT['B,OutSeqStep,EmbedLen'] = None
        self.past_values: FloatTensorT['B,OutSeqStep,EmbedLen'] = None


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
