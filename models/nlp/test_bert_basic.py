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

    def test_input_embedding(self):
        d_model = 512
        seq_len_max = 2000
        src_vocab = 2
        dropout_prob = .1

        position = partial(PositionalEncoding, seq_len_max, d_model,
                           dropout_prob)
        word_to_ix = {"hello": 0, "world": 1}
        embeds = nn.Embedding(src_vocab, d_model)

        lookup_tensor = torch.tensor([word_to_ix["hello"]], dtype=torch.long)
        hello_embed = embeds(lookup_tensor)
        self.assertEqual(hello_embed.shape, (1, d_model))

        lookup_tensor = torch.tensor([word_to_ix["hello"], word_to_ix["world"]],
                                     dtype=torch.long)
        sentence_embedding = embeds(lookup_tensor)
        self.assertEqual(sentence_embedding.shape, (2, d_model))

        lookup_tensor = torch.tensor([word_to_ix["hello"], 0], dtype=torch.long)
        hellos_embed = embeds(lookup_tensor)
        self.assertTrue((hellos_embed[0, :] == hellos_embed[1, :]).all())

        # include batch dim, hence the extra square brackets on the outside
        batch_lookup_tensor = torch.tensor(
            [[word_to_ix["hello"], word_to_ix["world"]]],
            dtype=torch.long)
        seq_length = batch_lookup_tensor.size(1)  # 2
        w_batch_embedding = embeds(batch_lookup_tensor)
        self.assertEqual(w_batch_embedding.shape,
                         (1, seq_length, d_model))  # 1, 2, 512

        embed_head = nn.Sequential(Embeddings(src_vocab, d_model), position())
        position_embedding = embed_head(batch_lookup_tensor)
        self.assertEqual(position_embedding.shape,
                         (1, seq_length, d_model))  # 1, 2, 512

    # then later add
    #
    # def test output_enmbedding():
    #     query = torch.Tensor([[[1, 1, 1], [0, 0, 0]]])
    #     key = torch.Tensor([[[1, 1, 1], [0, 0, 0]]])
    #     value = torch.Tensor([[[2, 2, 2], [1, 1, 1]]])
    #     ans, p = attention(query, key, value)
    #     assert ans.shape == (1, 2, 3)
    #     torch.testing.assert_close(ans.to(torch.float16), torch.Tensor(
    #         [[[1.8497, 1.8497, 1.8497], [1.5, 1.5, 1.5]]]).to(torch.float16))

    # def test_forward():
    #     assert False
    #
    #
    # def test_encode():
    #     assert False
    #
    #
    # def test_decode():
    #     assert False

    def test_attention(self):
        query = torch.Tensor([[[1, 1, 1], [0, 0, 0]]])
        key = torch.Tensor([[[1, 1, 1], [0, 0, 0]]])
        value = torch.Tensor([[[2, 2, 2], [1, 1, 1]]])
        ans, p = attention(query, key, value)
        self.assertEqual(ans.shape, (1, 2, 3))
        torch.testing.assert_close(ans.to(torch.float16), torch.Tensor(
            [[[1.8497, 1.8497, 1.8497], [1.5, 1.5, 1.5]]]).to(torch.float16))

    def test_mask(self):
        x = torch.ones(2, 3, 2)
        mask = torch.Tensor([
            [[1, 1]], [[0, 1]]
        ])
        self.assertEqual(mask.shape, (2, 1, 2))
        prediction = x.masked_fill(mask == 0, 0)
        target = torch.tensor([
            [[1., 1.], [1., 1.], [1., 1.]],
            [[0., 1.], [0., 1.], [0., 1.]]
        ])
        torch.testing.assert_close(prediction, target)

    def test_inference(self):
        for _ in range(1):
            test_model = make_model(src_vocab=11, tgt_vocab=11, N=6,
                                    d_model=512)
            test_model.eval()
            # one_sentence = LongTensorT([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]],
            #                            dtype=torch.long)
            one_sentence = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
            n_batch, seq_len = one_sentence.shape  # 1, 10
            assert one_sentence.shape == (n_batch, seq_len)

            src_mask = torch.ones(1, 1, 10)

            memory = test_model.encode(one_sentence, src_mask)
            ys = torch.zeros(1, 1).type_as(one_sentence)

            for i in range(9):
                self.assertEqual(memory.shape, (1, 10, 512))
                self.assertEqual(ys.shape, (1, 1 + i))
                out = test_model.decode(
                    memory, src_mask, ys,
                    subsequent_mask(ys.size(1)).type_as(one_sentence.data)
                )
                prob = test_model.generator(out[:, -1])
                _, next_word = torch.max(prob, dim=1)
                next_word = next_word.data[0]
                self.assertEqual(next_word.shape, ())
                ys = torch.cat(
                    [ys,
                     torch.empty(1, 1).type_as(one_sentence.data).fill_(
                         next_word)],
                    dim=1
                )

            print("Example Untrained Model Prediction:", ys)
