import copy
import math

import torch.nn as nn
from torch.nn.functional import log_softmax

from mgz.typing import *


# import altair as alt
# import GPUtil


class BaseTransformer(nn.Module):

    def forward(self, src_ids: LongTensorT['B,SrcSeqLen'],
                tgt_ids: LongTensorT['B,OutSeqLen'],
                src_mask: IntTensorT['B,SrcSeqLen'],
                tgt_mask: IntTensorT['B,OutSeqLen']):
       raise NotImplementedError

    def encode(self, src: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def decode_train(self, memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
                     src_mask: IntTensorT['B,OutSeqLen'],
                     tgt: LongTensorT['B,OutSeqLen'], tgt_mask):
        raise NotImplementedError

    def decode(self, memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
               src_mask: IntTensorT['B,OutSeqLen'],
               tgt: IntTensorT['B,OutSeqLen'],
               tgt_mask: IntTensorT['1,OutSeqLen']):
        raise NotImplementedError


