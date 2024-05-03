from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

import unittest

from transformers import LlamaTokenizer

from mgz.ds.sentence_datasets.gpt_input_augments import DocumentRequestChat, \
    PromptType, PromptConfig
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask, strings_to_padded_id_tensor_w_mask


class TestBert(unittest.TestCase):

    def test_prompts_to_padded_id_tensor_w_mask_NonTruncated(self):
        tokenizer = LlamaTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        prompt_cfg: PromptConfig = PromptConfig(tokenizer)
        prompt1 = DocumentRequestChat(prompt_config=prompt_cfg,
                                      document_text="document text",
                                      document_requests="tag")
        prompt2 = DocumentRequestChat(prompt_config=prompt_cfg,
                                      document_text="document text hello hello hello",
                                      document_requests="tag")
        src_ids_short_ref, src_mask_short_ref = strings_to_padded_id_tensor_w_mask(
            [prompt1.get_tokenizer_input(False),
             prompt2.get_tokenizer_input(False)],
            tokenizer=tokenizer,
            max_len=100, device='cpu')
        src_ids_short, src_mask_short = prompts_to_padded_id_tensor_w_mask(
            [prompt1, prompt2],
            tokenizer=tokenizer,
            max_len=100, device='cpu')
        reference_decoded = tokenizer.batch_decode(src_ids_short_ref,
                                                   skip_special_tokens=True)

        prompt_cutting_decoded = tokenizer.batch_decode(src_ids_short,
                                                        skip_special_tokens=True)
        print('-')
        print(prompt1.get_tokenizer_input(False))
        print('-')
        print(prompt1.get_tokenizer_input(True))
        print('-')
        print('ref_ids_short', reference_decoded)
        print('src_ids_short', prompt_cutting_decoded)
        print('src_mask_short_ref', tokenizer.batch_decode(src_mask_short_ref,
                                                           skip_special_tokens=True))
        print('src_mask_short',
              tokenizer.batch_decode(src_mask_short, skip_special_tokens=True))
        for i in range(len(reference_decoded)):
            self.assertTrue(reference_decoded[i] == prompt_cutting_decoded[i])
        decoded = tokenizer.batch_decode(src_ids_short[0],
                                         skip_special_tokens=True)
        ref_decoded = tokenizer.batch_decode(src_ids_short_ref[0],
                                             skip_special_tokens=True)
        print(len(decoded))
        print(decoded)
        print(len(ref_decoded))
        print(ref_decoded)
        self.assertTrue((src_mask_short == src_mask_short_ref).all())

    def test_prompts_to_padded_id_tensor_w_mask_Truncated(self):
        tokenizer = LlamaTokenizer.from_pretrained('mistralai/Mistral-7B-Instruct-v0.1')
        tokenizer.pad_token_id = tokenizer.eos_token_id
        sample_text = "we are going to make sure that this is correctly truncated"
        prompt_cfg: PromptConfig = PromptConfig(prompt_type=(PromptType.ADAPT))
        prompt_to_truncate = DocumentRequestChat(prompt_config=prompt_cfg,
                                                 document_text=sample_text + " truncate this",
                                                 document_requests="tag")

        prompt_to_truncate2 = DocumentRequestChat(prompt_config=prompt_cfg,
                                                  document_text=sample_text + " truncate truncate this",
                                                  document_requests="tag")

        src_ids_short, src_mask_short = prompts_to_padded_id_tensor_w_mask(
            [prompt_to_truncate, prompt_to_truncate2],
            tokenizer=tokenizer,
            max_len=72, device='cpu')
        print('src_ids_short',
              tokenizer.batch_decode(src_ids_short, skip_special_tokens=True))
        print('src_mask_short',
              tokenizer.batch_decode(src_mask_short, skip_special_tokens=True))
        decoded = tokenizer.batch_decode(src_ids_short,
                                         skip_special_tokens=True)
        self.assertTrue(sample_text in decoded[0])
        self.assertFalse("truncate this" in decoded[0])
        self.assertTrue("[/INST]" in decoded[0])
