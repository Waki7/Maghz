from __future__ import annotations

import unittest
from functools import partial

import torch.nn as nn

from mgz.models.metrics import *

#
# MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT

# import altair as alt
# import GPUtil

from mgz.models.nlp.bart_orig import BartAttention
from mgz.models.nlp.bart import MultiHeadedAttention


class TestBert(unittest.TestCase):
    def setUp(self):
        pass

    def test_relative_error(self):
        worse_corpus = [['My', 'full', 'pytorch', 'test'],
                        ['Another', 'Sentence']]
        better_corpus = [['My', 'full', 'pytorch', 'test'],
                         ['No', 'Match']]
        references_corpus = [
            [['My', 'full', 'pytorch', 'test'],
             ['Completely', 'Different']],
            [['No', 'Match']]
        ]
        score_worse = bleu(worse_corpus, references_corpus)
        score_better = bleu(better_corpus, references_corpus)
        self.assertGreater(score_better, score_worse)
        modified_corpus = [
            [['My', 'full', 'pytorch', 'test'],
             ['Completely', 'Different']],
            [['No', 'Match']]
        ]
        self.assertEqual(score_better, bleu(better_corpus, modified_corpus))
