from __future__ import annotations

import unittest
from mgz.models.nlp.led import LEDForBinaryTagging
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

        tokenizer_local = LEDForBinaryTagging.load_tokenizer(local_path)
        tokenizer_hug = LEDForBinaryTagging.load_tokenizer(tokenizer_name)
        self.recursive_check(tokenizer_local, tokenizer_hug)
