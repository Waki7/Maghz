from mgz.ds import SentenceDataset
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask


class Sent2SentBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: LongTensorT['B,SrcSeqLen'],
                 tgt: LongTensorT['B,TgtSeqLen'] = None,
                 pad_idx=2):
        self.src: LongTensorT['B,SrcSeqLen'] = src
        self.src_mask: IntTensorT['B, SrcSeqLen'] = (
                src != pad_idx).long()
        if tgt is not None:
            self.tgt: LongTensorT['B,TgtSeqLen'] = tgt
            self.tgt_mask: LongTensorT[
                'B,TgtSeqLen,SrcSeqLen'] = self.make_gen_mask(self.tgt, pad_idx)
            self.ntokens: int = (self.tgt != pad_idx).data.sum()
        self.tgt_mask.to(self.tgt.device)
        self.src_mask.to(self.src.device)

    @staticmethod
    def make_gen_mask(tgt: LongTensorT['B,SrcSeqLen'], pad: int) -> \
            LongTensorT['B,TgtSeqLen,SrcSeqLen']:
        "Create a mask to hide padding and future words. This is only needed"
        "when training for generation."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return LongTensorT(tgt_mask)

    @staticmethod
    def collate_batch(
            batch: List[Tuple[SrcStringT, TgtStringT]],
            src_tokenizer: PreTrainedTokenizerBase,
            tgt_tokenizer: PreTrainedTokenizerBase,
            device,
            max_src_len: int,
            max_tgt_len: int
    ) -> Sent2SentBatch:
        srcs: List[SrcStringT]
        tgts: List[TgtStringT]
        srcs, tgts = zip(*batch)
        src: LongTensorT['B,SrcSeqLen'] = \
            strings_to_padded_id_tensor(srcs,
                                        tokenizer=src_tokenizer,
                                        max_len=max_src_len,
                                        device=device)
        tgt: LongTensorT['B,TgtSeqLen'] = \
            strings_to_padded_id_tensor(tgts,
                                        tokenizer=tgt_tokenizer,
                                        max_len=max_tgt_len,
                                        device=device)

        assert src_tokenizer.pad_token_id == tgt_tokenizer.pad_token_id, \
            "Pad tokens must be the same for src and tgt"
        return Sent2SentBatch(src, tgt, pad_idx=src_tokenizer.pad_token_id)

    @staticmethod
    def default_collate_fn(ds: SentenceDataset,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcStringT, TgtStringT]]):
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"

        return Sent2SentBatch.collate_batch(
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
