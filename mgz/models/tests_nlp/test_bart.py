from __future__ import annotations

import unittest
from functools import partial

import torch.nn as nn

from mgz.models.nlp.bert_basic import Embeddings, attention, PositionalEncoding, \
    make_model, subsequent_mask
from mgz.typing import *

#
# MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT

# import altair as alt
# import GPUtil

from mgz.models.nlp.bart_orig import BartAttention
from mgz.models.nlp.bart import MultiHeadedAttention


class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def verify_attention(self):
        attn_orig = BartAttention(512, 8, 0.1)
        attn_new = MultiHeadedAttention(8, 512, 0.1)
