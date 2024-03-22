from mgz.ds import SentenceDataset
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask, \
    prompts_to_padded_id_tensor_w_mask

'''

Model

Dataset

    Batch Implementation of Dataset
    Batch




'''
class BehavioralBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self,
                 src_ids: LongTensorT['TaskSize,SrcSeqLen'],
                 src_masks: IntTensorT['TaskSize,SrcSeqLen']):
        self.src_ids: LongTensorT['TaskSize,SrcSeqLen'] = src_ids
        self.src_masks: IntTensorT['TaskSize,SrcSeqLen'] = src_masks

    @staticmethod
    def collate_batch(
            batch: List[Tuple[SrcStringT, TgtStringT]],
            tokenizer: PreTrainedTokenizerBase,
            device,
            max_src_len: int,
            max_tgt_len: int
    ) -> BehavioralBatch:

        assert max_tgt_len is None, "max_tgt_len must not be set for this task"
        prompts, labels = zip(*batch)
        src_ids, src_masks = \
            prompts_to_padded_id_tensor_w_mask(prompts=prompts,
                                               tokenizer=tokenizer,
                                               max_len=max_src_len,
                                               device=device)
        label_tensor = LongTensorT(
            torch.tensor(labels, dtype=torch.long, device=device))
        return BehavioralBatch(src_ids=src_ids,
                                  src_masks=src_masks,
                                  labels=label_tensor)

    @staticmethod
    def default_collate_fn(ds: SentenceDataset,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcStringT, TgtStringT]]):
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        return BehavioralBatch.collate_batch(
            batch=batch,
            src_tokenizer=ds.tokenizer_src,
            tgt_tokenizer=ds.tokenizer_tgt,
            device=device,
            max_src_len=ds.max_src_len,
            max_tgt_len=ds.max_tgt_len
        )


from __future__ import annotations

import torch.utils.data
from transformers import PreTrainedTokenizerBase

from mgz.ds.base_dataset import DataState
from mgz.typing import *
