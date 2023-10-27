from __future__ import annotations

import unittest

import transformers as hug

from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *
from settings import DEVICE


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

    def test_gather(self):  # mps bug
        # t = torch.LongTensor([[0, 0], [1, 1], [2, 2]]).to(DEVICE)
        # print(t.shape)
        # indices = torch.LongTensor([[0, 0], [2, 2], [1, 1]]).to(DEVICE)
        # print(indices.shape)
        t = torch.LongTensor([0, 1, 2]).view(-1, 1, 1).to(DEVICE)
        indices = torch.LongTensor([0, 1, 2]).view(-1, 1, 1).to(DEVICE)
        gathered = t.gather(0, indices)
        print(gathered)

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

    def test_quantization(self):
        import torch
        import torch.nn as nn

        from bitsandbytes.nn import Linear8bitLt
        fp16_model = nn.Sequential(
            nn.Linear(64, 64),
            nn.Linear(64, 64)
        )
        torch.save(fp16_model.state_dict(), "./test_data/model.pt")

        int8_model = nn.Sequential(
            Linear8bitLt(64, 64, has_fp16_weights=False),
            Linear8bitLt(64, 64, has_fp16_weights=False)
        )
        int8_model.load_state_dict(torch.load("./test_data/model.pt"))
        int8_model = int8_model.to(0)  # Quantization happens here
        pre_weight = int8_model[0].weight

        # storing 8bit model
        torch.save(int8_model.state_dict(), "./test_data/model_int8.pt")

        # can't load 8 bit into 16 bit model
        int8_model.load_state_dict(torch.load("./test_data/model_int8.pt"))
        post_weight = int8_model[0].weight
        # note how we didn't have to do the .to(0) again, only quantized once
        self.assertTrue(torch.all(pre_weight == post_weight))
