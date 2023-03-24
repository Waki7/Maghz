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


class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def verify_attention(self):
        pass