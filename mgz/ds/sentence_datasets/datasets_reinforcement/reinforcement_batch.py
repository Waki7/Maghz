from __future__ import annotations

import torch.utils.data
from transformers import PreTrainedTokenizerBase

from mgz.ds import SentenceDataset
from mgz.ds.base_dataset import DataState
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask
from mgz.ds.sentence_datasets.gpt_input_augments import BatchChatInput
from mgz.typing import *


class ReinforcementBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_masks: IntTensorT['B,SrcSeqLen'],
                 original_lengths: IntTensorT['B'],
                 reward_function: Callable[[List[str]], List[float]] = None):
        self.src_ids: LongTensorT['B,SrcSeqLen'] = src_ids
        self.src_masks: IntTensorT['B,SrcSeqLen'] = src_masks
        self.original_lengths: IntTensorT['B'] = original_lengths
        # self.reward_function: Callable[
        #     [List[str]], List[float]] = self.reward_function()

    def reward_function(self, predictions: List[str]) -> List[float]:
        return [1 if 'yes' == pred[0].lower() or 'no' == pred[0].lower() else -1
                for pred in predictions]

    def get_reward(self, sampled_tokens: LongTensorT['B,TgtSeqLen'],
                   tokenizer) -> FloatTensorT['B']:
        predictions = tokenizer.batch_decode(sampled_tokens)
        return FloatTensorT(self.reward_function(predictions)).to(
            self.src_ids.device)

    @staticmethod
    def collate_batch(
            batch: List[BatchChatInput],
            tokenizer: PreTrainedTokenizerBase,
            device,
            max_src_len: int,
            max_tgt_len: int
    ) -> ReinforcementBatch:
        assert max_tgt_len is None, "max_tgt_len must not be set for this task"
        src_ids, src_masks, original_lengths = \
            prompts_to_padded_id_tensor_w_mask(prompts=batch,
                                               tokenizer=tokenizer,
                                               max_len=max_src_len,
                                               device=device,
                                               return_original_lengths=True)

        return ReinforcementBatch(src_ids=src_ids,
                                  src_masks=src_masks,
                                  original_lengths=original_lengths)

    @staticmethod
    def default_collate_fn(ds: SentenceDataset,
                           device: Union[int, torch.device],
                           batch: List[BatchChatInput]) -> ReinforcementBatch:
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        return ReinforcementBatch.collate_batch(
            batch=batch,
            tokenizer=ds.tokenizer_tgt,
            device=device,
            max_src_len=ds.max_src_len,
            max_tgt_len=ds.max_tgt_len
        )
