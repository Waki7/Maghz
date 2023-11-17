from __future__ import annotations

import unittest

import torch
import transformers as hug

import mgz.settings as settings
from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.version_control import lookup_or_init_model


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

        tokenizer_local = LEDForConditionalGeneration.load_tokenizer(
            local_path)
        tokenizer_hug = LEDForConditionalGeneration.load_tokenizer(
            local_path)
        self.recursive_check(tokenizer_local, tokenizer_hug)

    def test_against_huggingface(self):
        tokenizer = hug.LlamaTokenizerFast.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        # tokenizer = hug.LEDTokenizerFast.from_pretrained(
        #     "allenai/led-base-16384-multi_lexsum-source-long")
        x = tokenizer("hello world", return_tensors="pt").input_ids
        self.assertEqual(tokenizer.pad_token_id, 1)
        self.assertEqual(tokenizer.sep_token_id, 2)
        self.assertEqual(tokenizer.eos_token_id, 2)
        self.assertEqual(tokenizer.bos_token_id, 0)
        self.assertEqual(tokenizer.unk_token_id, 3)
        self.assertEqual(1, 0)

        hug_model = hug.MistralForSequenceClassification.from_pretrained(
            "mistralai/Mistral-7B-v0.1")

        expected_logits = hug_model(x).logits
        del hug_model

        mgz_model = LEDForConditionalGeneration.from_pretrained(
            "mistralai/Mistral-7B-v0.1")

    def test_mistral_tokenizer(self):
        test_string = ['This is a test',
                       'This is another test, adding words.']

        tex_len = 10
        bsz = 2
        assert len(test_string) == bsz

        tokenizer = hug.LlamaTokenizerFast.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        config = hug.MistralConfig.from_pretrained(
            "mistralai/Mistral-7B-v0.1")
        print('pad_token_id', config.pad_token_id)
        print('sep_token_id', config.sep_token_id)
        print('sep_token_id', config.eos_token_id)
        tokenizer.pad_token_id = tokenizer.eos_token_id
        tokenizer.sep_token_id = tokenizer.eos_token_id
        print('pad_token_id', tokenizer.pad_token_id)
        print('sep_token_id', tokenizer.sep_token_id)
        print('eos_token_id', tokenizer.eos_token_id)
        print('bos_token_id', tokenizer.bos_token_id)
        print('unk_token_id', tokenizer.unk_token_id)
        input_ids = tokenizer.__call__(test_string, return_tensors="pt",
                                       padding=True, max_length=tex_len,
                                       truncation=False).input_ids
        print(input_ids)
        if config.pad_token_id is None:
            sequence_lengths = -1
        else:
            if input_ids is not None:
                sequence_lengths = (torch.eq(input_ids,
                                             config.pad_token_id).long().argmax(
                    -1) - 1)
            else:
                sequence_lengths = -1
        print('seq len', (torch.eq(input_ids,
                                   tokenizer.pad_token_id).long().argmax(
            -1) - 1))
        print('seq len', sequence_lengths)
        logits = torch.randn(2, 10, 2)
        print(logits)
        pooled_logits = logits[
            torch.arange(2, device=logits.device), sequence_lengths]
        print(pooled_logits)

    def test_hug_mistral(self):
        from transformers import LlamaTokenizerFast
        from mgz.models.nlp.mistral_orig import MistralForCausalLM
        model = MistralForCausalLM.from_pretrained(
            "mistralai/Mistral-7B-v0.1", use_flash_attention_2=True,
            torch_dtype=torch.float16, device_map={"": settings.DEVICE}, )
        tokenizer = LlamaTokenizerFast.from_pretrained(
            "mistralai/Mistral-7B-v0.1")

        prompt = "My favourite condiment is"

        model_inputs = tokenizer([prompt], return_tensors="pt").to(
            settings.DEVICE)
        # model.config.max_length = 3
        print(model.config.max_length)
        generated_ids = model.generate(**model_inputs, max_new_tokens=100)
        print(tokenizer.batch_decode(generated_ids)[0])
        
    def test_my_mistral(self):
        from mgz.models.nlp.mistral import MistralForCausalLM
        model_node = lookup_or_init_model(MistralForCausalLM,
                                          "mistralai/Mistral-7B-v0.1",
                                          "mistralai/Mistral-7B-v0.1")
        model = model_node.model.half()
        tokenizer = model_node.tokenizer

        prompt = "My favourite condiment is"

        model_inputs = tokenizer([prompt], return_tensors="pt").to(
            settings.DEVICE)

        # model.config.max_length = 7
        generated_ids = model.generate(src_ids=model_inputs.input_ids,
                                       src_mask=model_inputs.attention_mask,
                                       tgt_ids=model_inputs.input_ids,
                                       max_new_tokens=100)
        print(tokenizer.batch_decode(generated_ids)[0])
