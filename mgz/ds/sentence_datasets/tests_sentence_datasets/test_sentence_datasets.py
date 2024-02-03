from __future__ import annotations
from __future__ import annotations
from __future__ import annotations

import unittest

from transformers import LlamaTokenizer

from mgz.ds.sentence_datasets.gpt_input_augments import ContextPromptingInput, \
    PromptType
from mgz.ds.sentence_datasets.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask, strings_to_padded_id_tensor_w_mask


class TestBert(unittest.TestCase):

    def test_prompts_to_padded_id_tensor_w_mask_NonTruncated(self):
        tokenizer = LlamaTokenizer.from_pretrained('AdaptLLM/law-chat')
        prompt1 = ContextPromptingInput(PromptType.ADAPT, "document_text",
                                        "tag")
        prompt2 = ContextPromptingInput(PromptType.ADAPT,
                                        "document_text hello hello hello",
                                        "tag")
        src_ids_short_ref, src_mask_short_ref = strings_to_padded_id_tensor_w_mask(
            [prompt1.get_tokenizer_input(False),
             prompt2.get_tokenizer_input(False)],
            tokenizer=tokenizer,
            max_len=100, device='cpu')
        src_ids_short, src_mask_short = prompts_to_padded_id_tensor_w_mask(
            [prompt1, prompt2],
            tokenizer=tokenizer,
            max_len=100, device='cpu')
        print('src_ids_short_ref', tokenizer.batch_decode(src_ids_short_ref,
                                                          skip_special_tokens=True))
        print('src_ids_short',
              tokenizer.batch_decode(src_ids_short, skip_special_tokens=True))

        print('src_mask_short_ref', tokenizer.batch_decode(src_mask_short_ref,
                                                           skip_special_tokens=True))
        print('src_mask_short',
              tokenizer.batch_decode(src_mask_short, skip_special_tokens=True))
        self.assertTrue((src_ids_short == src_ids_short_ref).all())
        self.assertTrue((src_mask_short == src_mask_short_ref).all())

    def test_prompts_to_padded_id_tensor_w_mask_Truncated(self):
        tokenizer = LlamaTokenizer.from_pretrained('AdaptLLM/law-chat')
        sample_text = "we are going to make sure that this is correctly truncated"
        prompt_to_truncate = ContextPromptingInput(PromptType.ADAPT,
                                                   sample_text + " truncate this",
                                                   "tag")
        src_ids_short, src_mask_short = prompts_to_padded_id_tensor_w_mask(
            [prompt_to_truncate],
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
