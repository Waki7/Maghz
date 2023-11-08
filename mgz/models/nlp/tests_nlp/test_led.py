from __future__ import annotations

import os
import unittest

import transformers as hug

from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.typing import *
from mgz.version_control.model_index import get_models_path


class TestLED(unittest.TestCase):
    def recursive_check(self, obj1, obj2):
        for key, val in vars(obj1).items():
            if hasattr(val, "__dict__"):
                self.assertTrue(hasattr(obj2, key))
                self.recursive_check(val, getattr(obj2, key))
            else:
                self.assertTrue(hasattr(obj2, key))
                self.assertEqual(val, getattr(obj2, key), msg=val)

    def setUp(self):
        model_path = os.path.join(get_models_path(),
                                  'allenai/led-base-16384-multi_lexsum-source-long')
        if not os.path.exists(os.path.join(model_path, 'weights.bin')):
            LEDForConditionalGeneration.initial_save(
                'allenai/led-base-16384-multi_lexsum-source-long',
                get_models_path())
        self.model = LEDForConditionalGeneration.load_model(model_path).cpu()
        self.config = self.model.config

        self.hug_model = hug.LEDForSequenceClassification(
            self.config).cpu()
        self.hug_model.load_state_dict(self.model.state_dict())

        self.tokenizer = LEDForConditionalGeneration.load_tokenizer(model_path)

    # ------------------ TESTING LOADING ------------------
    def test_attention(self):
        local_path = os.path.join(get_models_path(),
                                  'allenai/led-base-16384-multi_lexsum-source-long')
        tokenizer_name = 'allenai/led-base-16384-multi_lexsum-source-long'

        tokenizer_local = LEDForConditionalGeneration.load_tokenizer(
            local_path)
        tokenizer_hug = LEDForConditionalGeneration.load_tokenizer(
            local_path)
        self.recursive_check(tokenizer_local, tokenizer_hug)

    def test_sequence_classification(self):
        test_string = ['This is a bad test',
                       'This is a good test, more words.']
        tag_string = ['bad', 'good']
        max_src_len = 10
        max_tgt_len = 2

        src_encodings = self.tokenizer(test_string, return_tensors="pt",
                                       max_length=max_src_len, truncation=False,
                                       padding=True)
        src_ids: LongTensorT['B,SrcSeqLen'] = src_encodings.input_ids
        src_mask = src_encodings.attention_mask

        tgt_encodings = self.tokenizer(tag_string, return_tensors="pt",
                                       max_length=max_tgt_len, truncation=False,
                                       padding=True)
        tgt_ids: LongTensorT['EmbedLen'] = tgt_encodings.input_ids
        tgt_mask = tgt_encodings.attention_mask

        print('src_ids.shape', src_ids.shape)
        print('tgt_ids.shape', tgt_ids.shape)
        # don't need tgt_mask because you are generating one token at a time
        a = self.model.forward(src_ids=src_ids, tgt_ids=tgt_ids,
                               src_mask=src_mask, tgt_mask=tgt_mask)
