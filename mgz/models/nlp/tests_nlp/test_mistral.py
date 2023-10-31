from __future__ import annotations

import unittest

from mgz.ds.sentence_datasets.aus_legal_case_reports import \
    AusCaseReportsToTagGrouped
from mgz.models.nlp.led import LEDForSequenceClassification
import transformers as hug
from tokenizers import Tokenizer


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
        pass

    # ------------------ TESTING LOADING ------------------
    def test_attention(self):
        local_path = 'allenai/led-base-16384-multi_lexsum-source-long'
        tokenizer_name = 'allenai/led-base-16384-multi_lexsum-source-long'

        tokenizer_local = LEDForSequenceClassification.load_tokenizer(
            local_path)
        tokenizer_hug = LEDForSequenceClassification.load_tokenizer(
            tokenizer_name)
        self.recursive_check(tokenizer_local, tokenizer_hug)

    def test_against_huggingface(self):
        hug_model = hug.MistralForSequenceClassification.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        tokenizer = hug.AutoTokenizer.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        x = tokenizer("hello world", return_tensors="pt").input_ids
        print(x.shape)
        # hug_model.forward(x)
        self.assertEqual(1, 0)
