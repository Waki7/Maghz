from __future__ import annotations

import unittest
from functools import partial

import torch
import torch.nn as nn

from mgz.models.nlp.base_transformer import BaseTransformer, \
    BeamInference, \
    LogitsRuleEnforce
from mgz.typing import *
from settings import DEVICE

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

    def check_tensor(self, prediction, expected):
        self.assertTrue((prediction == expected).all(),
                        'predicted \n{}\n not equal to expected \n{}\n'.format(
                            prediction, expected))

    def test_combine_dims_repeat(self):
        tensor = torch.randn([2, 3, 4])
        rpt1 = tensor.unsqueeze(0).repeat(4, 1, 1, 1)
        rpt2 = tensor.repeat(4, 1, 1).view(-1, 2, 3, 4)
        self.assertTrue((rpt1 == rpt2).all())

    def test_logits_rules(self):
        logits = torch.Tensor([
            [[0.1, 0.2, 0.3, 0.4]],
            [[0.1, 0.2, 0.3, 0.4]],
        ]).to(DEVICE)
        self.assertEqual(logits.shape, (2, 1, 4))
        batch_size = 1
        beam_size = int(logits.shape[0] / batch_size)  # 2

        beam_ctx = BeamInference(beam_size)

        # testing after step one, when there's been a previous token
        beam_ctx.pred_tokens.append(torch.tensor([[3], [0]]))
        self.assertEqual(beam_ctx.pred_tokens[0].shape, (beam_size, batch_size))

        logits_rules = LogitsRuleEnforce(max_length=3, eos_token_id=1)
        expected = torch.Tensor([
            [0.1, -float('inf'), 0.3, -float('inf')],
            [-float('inf'), -float('inf'), 0.3, 0.4],
        ]).to(DEVICE)
        filtered_logits = logits_rules.__call__(beam_ctx=beam_ctx,
                                                new_logits=logits)
        print(filtered_logits)
        self.assertTrue((filtered_logits == expected).all(),
                        'filtered logits \n{}\n not equal to expected \n{}\n'.format(
                            filtered_logits, expected))

    def test_gather(self):  # mps bug
        # t = torch.LongTensor([[0, 0], [1, 1], [2, 2]]).to(DEVICE)
        # print(t.shape)
        # indices = torch.LongTensor([[0, 0], [2, 2], [1, 1]]).to(DEVICE)
        # print(indices.shape)
        t = torch.LongTensor([0, 1, 2]).view(-1, 1, 1).to(DEVICE)
        indices = torch.LongTensor([0, 1, 2]).view(-1, 1, 1).to(DEVICE)
        gathered = t.gather(0, indices)
        print(gathered)

    def test_beam_complete(self):
        n_beams = 3
        batch_sz = 2
        token_shp = (3, 2)
        beam_ctx = BeamInference(3, 2, 5)
        first_probs = torch.Tensor([
            [[-.01, -1.0, -1.0], [-1.0, -1.0, -.01]],
            [[-.01, -1.0, -1.0], [-1.0, -1.0, -.01]],
            [[-.01, -1.0, -1.0], [-1.0, -1.0, -.01]]
        ]).to(DEVICE).contiguous().view(n_beams * batch_sz, -1)
        assert first_probs.shape == (6, 3)  # 6 = 2 * 3
        tokens1 = beam_ctx.select_ids_from_logprobs(first_probs)
        expct_tokens1 = torch.Tensor([[0, 2],
                                      [1, 0],
                                      [2, 1]]).to(DEVICE)
        expct_indices1 = torch.Tensor([[0, 0],
                                       [0, 0],
                                       [0, 0]]).to(DEVICE)
        self.check_tensor(tokens1.view(token_shp), expct_tokens1)
        self.check_tensor(beam_ctx.beam_indices[-1].view(token_shp),
                          expct_indices1)
        self.assertTrue(not beam_ctx.is_done)

        second_probs = torch.Tensor([
            [[-0.3, -1.0, 1.0], [-0.2, -0.4, -1.0]],
            [[-1.0, -1.0, -0.2], [-0.2, -0.4, -0.3]],
            [[-0.1, -0.5, -1.0], [-1.0, -0.1, -1.0]]
        ]).to(DEVICE).contiguous().view(n_beams * batch_sz, -1)
        expct_tokens2 = torch.Tensor([[2, 2],
                                      [0, 1],
                                      [2, 0]]).to(DEVICE)
        expct_indices2 = torch.Tensor([[0, 0],
                                       [2, 2],
                                       [1, 1]]).to(DEVICE)
        tokens2 = beam_ctx.select_ids_from_logprobs(second_probs)
        self.check_tensor(tokens2.view(token_shp), expct_tokens2)
        self.check_tensor(beam_ctx.beam_indices[-1].view(token_shp),
                          expct_indices2)
        self.assertTrue(not beam_ctx.is_done)

        third_probs = torch.Tensor([
            [[-0.5, -1.0, -0.1], [-0.5, -0.7, -0.1]],
            [[-1.0, -1.0, -0.1], [-0.5, -0.7, -0.2]],
            [[-0.3, -0.7, -0.2], [-1.0, -0.4, -0.2]]
        ]).to(DEVICE).contiguous().view(n_beams * batch_sz, -1)
        expct_tokens3 = torch.Tensor([[2, 2],
                                      [2, 2],
                                      [2, 2]]).to(DEVICE)
        expct_indices3 = torch.Tensor([[0, 0],
                                       [1, 1],
                                       [2, 2]]).to(DEVICE)
        tokens3 = beam_ctx.select_ids_from_logprobs(third_probs)
        self.check_tensor(tokens3.view(token_shp), expct_tokens3)
        self.check_tensor(beam_ctx.beam_indices[-1].view(token_shp),
                          expct_indices3)
        self.assertTrue(beam_ctx.is_done)

    def test_beam_select_best(self):
        indices = [
            torch.LongTensor([0, 0, 0]).unsqueeze(-1).to(DEVICE),
            torch.LongTensor([1, 2, 2]).unsqueeze(-1).to(DEVICE),
            torch.LongTensor([2, 2, 1]).unsqueeze(-1).to(DEVICE),
        ]
        tokens = [
            torch.LongTensor([7, 4, 5]).unsqueeze(-1).to(DEVICE),
            torch.LongTensor([4, 3, 2]).unsqueeze(-1).to(DEVICE),
            torch.LongTensor([3, 5, 2]).unsqueeze(-1).to(DEVICE),
        ]
        scores = [
            torch.Tensor([.15, .2, .8]).unsqueeze(-1).to(DEVICE),
            torch.Tensor([.3, 1.2, 1.1]).unsqueeze(-1).to(DEVICE),
            torch.Tensor([1.4, 1.9, 1.7]).unsqueeze(-1).to(DEVICE),
        ]

        expected = torch.Tensor([5, 3, 2]).to(DEVICE)
        expected = torch.Tensor([5, 2, 5]).to(DEVICE)
        print('batch size is', indices[0].shape[-1])
        beam_ctx = BeamInference(3, 2, 5)
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