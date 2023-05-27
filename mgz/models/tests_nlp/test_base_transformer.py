from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn as nn

from mgz.models.nlp.base_transformer import BaseTransformer, BeamSearchContext, \
    LogitsRuleEnforce
from mgz.typing import *

import transformers as hug


class TestConfig(hug.PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = 2
        self.hidden_size = 512
        self.num_attention_heads = 8
        self.intermediate_size = 2048
        self.hidden_act = "gelu"
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.1
        self.max_position_embeddings = 2000
        self.type_vocab_size = 2
        self.initializer_range = 0.02


#
# MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT

# import altair as alt
# import GPUtil


class TestBert(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(3)

    def test_combine_dims_repeat(self):
        tensor = torch.randn([2, 3, 4])
        rpt1 = tensor.unsqueeze(0).repeat(4, 1, 1, 1)
        rpt2 = tensor.repeat(4, 1, 1).view(-1, 2, 3, 4)
        self.assertTrue((rpt1 == rpt2).all())

    def test_logits_rules(self):
        logits = torch.Tensor([
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.1, 0.2, 0.3, 0.4]],
        ])
        self.assertEqual(logits.shape, (2, 1, 4))
        batch_size = 1
        beam_size = int(logits.shape[0] / batch_size)  # 2

        beam_ctx = BeamSearchContext(beam_size)

        # testing after step one, when there's been a previous token
        beam_ctx.pred_tokens.append(torch.tensor([[3], [0]]))
        self.assertEqual(beam_ctx.pred_tokens[0].shape, (beam_size, batch_size))

        logits_rules = LogitsRuleEnforce(max_length=3, eos_token_id=1)
        expected = torch.Tensor([
            [0.1, -float('inf'), 0.3, -float('inf')],
            [-float('inf'), -float('inf'), 0.3, 0.4],
        ])
        filtered_logits = logits_rules.__call__(beam_ctx=beam_ctx,
                                                new_logits=logits)
        print(filtered_logits)
        self.assertTrue((filtered_logits == expected).all(),
                        'filtered logits \n{}\n not equal to expected \n{}\n'.format(
                            filtered_logits, expected))

    def test_beam_select_best(self):
        indices = [
            torch.LongTensor([0, 0, 0]).unsqueeze(-1),
            torch.LongTensor([1, 2, 0]).unsqueeze(-1),
            torch.LongTensor([2, 0, 1]).unsqueeze(-1),
        ]
        tokens = [
            torch.LongTensor([7, 4, 5]).unsqueeze(-1),
            torch.LongTensor([4, 3, 2]).unsqueeze(-1),
            torch.LongTensor([3, 5, 2]).unsqueeze(-1),
        ]
        scores = [
            torch.Tensor([.15, .2, .8]).unsqueeze(-1),
            torch.Tensor([.1, .4, 0.3]).unsqueeze(-1),
            torch.Tensor([.3, .8, .5]).unsqueeze(-1),
        ]

        expected = torch.Tensor([5, 3, 2])

        beam_ctx = BeamSearchContext(3)
        beam_ctx.pred_tokens = tokens
        beam_ctx.best_probs = scores
        beam_ctx.beam_indices = indices
        best_tokens = beam_ctx.get_best_sequence()
        self.assertTrue((best_tokens == expected).all(),
                        'filtered logits \n{}\n not equal to expected \n{}\n'.format(
                            best_tokens, expected))

    def test_beam_search(self):
        n_beams = 3
        b = 2
        seq_len = 1
        vocab_size = 5
        cfg = TestConfig(num_beams=n_beams)
        model = BaseTransformer(cfg)

        probs = torch.softmax(torch.randn([n_beams, b, seq_len, vocab_size]),
                              dim=-1)

        post_reshape = probs.permute(1, 2, 0, 3).flatten(2)
        self.assertEqual(post_reshape.shape, (b, seq_len, n_beams * vocab_size))
        best_probs, best_indices = \
            torch.topk(post_reshape, k=n_beams, dim=-1, sorted=True)
        best_probs: FloatTensorT['n_beams,b,seq_len'] = \
            best_probs.permute(2, 0, 1)
        best_indices: LongTensorT['n_beams,b,seq_len'] = \
            best_indices.permute(2, 0, 1)
        vocab_idx = best_indices % vocab_size
        beam_idx = best_indices // vocab_size
        expected_indices = torch.tensor([[[2], [1]], [[2], [4]], [[4], [2]]])
        print(best_probs.shape)
        print(beam_idx.shape)
        self.assertTrue((vocab_idx == expected_indices).all())
