# from __future__ import annotations
#
# import unittest
#
# from mgz.models.nlp.metrics import *
#
#
# #
# # MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT
#
# # import altair as alt
# # import GPUtil
#
#
# class TestBert(unittest.TestCase):
#     def setUp(self):
#         idxer = Indexer.get_default_index()
#
#         self.model, self.tokenizer = idxer.get_cached_runtime_nlp_model(
#             'facebook/bart-base',
#             BartModel)
#
#     def test_input_embedding(self):
#         tokenized1 = ['calderbank letter']
#         tokenized2 = ['calderbank letter']
#         print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))
#         print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))
#         print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))
#         self.assertTrue(False)
