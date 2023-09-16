from __future__ import annotations

import copy
from enum import Enum

import torch.utils.data
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer

from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.typing import *


class SampleType(Enum):
    # MultiLexSum keys
    ID = 'id'
    INPUT_TEXT = 'sources'
    SUMMARY_TINY = 'summary/tiny'
    SUMMARY_SHORT = 'summary/short'
    SUMMARY_LONG = 'summary/long'

    # Australian Legal Case Reports keys
    CATCHPHRASES = 'catchphrase'
    TAG = 'tag'
    NAME = 'name'
    KEY = 'key'
    # KEY = 'key'


def subsequent_mask(size: SrcSeqLen):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class Sent2TagBatch:
    # @TODO
    pass


class Sent2TagMetaTaskBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self,
                 per_tag_pos_srcs: List[LongTensorT['NShot,SrcSeqLen']],
                 per_tag_neg_srcs: List[LongTensorT['NShot,SrcSeqLen']],
                 tgt_tags: List[LongTensorT['TgtSeqLen']],
                 pad_idx=2):

        self.src_pos_masks = \
            [(p_src != pad_idx).long().to(p_src.device) for p_src in
             per_tag_pos_srcs]
        self.src_neg_masks = \
            [(n_src != pad_idx).long().to(n_src.device) for n_src in
             per_tag_neg_srcs]
        self.tgt_tag_masks = [(tgt != pad_idx).long().to(tgt.device) for tgt in
                              tgt_tags]

        self.per_tag_pos_srcs: List[
            LongTensorT['B,SrcSeqLen']] = per_tag_pos_srcs
        self.per_tag_neg_srcs: List[
            LongTensorT['B,SrcSeqLen']] = per_tag_neg_srcs
        self.tgt_tag = tgt_tags

    @staticmethod
    def collate_batch(
            tag_task: Dict[
                TgtStringT, Tuple[List[SrcStringT], List[SrcStringT]]],
            src_tokenizer_pipeline: Callable[[SrcStringT], List[str]],
            tgt_tokenizer_pipeline: Callable[[TgtStringT], List[str]],
            src_vocab_pipeline: Callable[List[str], List[int]],
            tgt_vocab_pipeline: Callable[List[str], List[int]],
            device,
            pad_id: int,
            max_src_len=128,
            max_tgt_len=128
    ) -> Sent2TagMetaTaskBatch:

        # ) -> Tuple[LongTensorT['B,SrcSeqLen'], LongTensorT['B,OutSeqLen']]:
        dtype = torch.int32
        bs_id = torch.tensor([0], device=device, dtype=dtype)  # <s> token id
        eos_id = torch.tensor([1], device=device, dtype=dtype)  # </s> token id
        _src: SrcStringT
        _tgt: TgtStringT

        per_tag_pos_srcs: List[LongTensorT['NShot,SrcSeqLen']] = []
        per_tag_neg_srcs: List[LongTensorT['NShot,SrcSeqLen']] = []
        tgt_tags: List[LongTensorT['TgtSeqLen']] = []
        for tag, pos_neg_srcs in tag_task.items():
            src_pos_list: List[torch.Tensor('SrcSeqLen')] = []  # SrcSeqLen
            src_neg_list: List[torch.Tensor('SrcSeqLen')] = []

            # At the moment it's enforced that srcs_pos and srcs_neg are of
            # the same length, but this may change
            srcs_pos: List[SrcStringT] = pos_neg_srcs[0]
            srcs_neg: List[SrcStringT] = pos_neg_srcs[1]

            for src_pos in srcs_pos:
                processed_src_pos: torch.Tensor('SrcSeqLen') = \
                    torch.cat([bs_id, torch.tensor(
                        src_vocab_pipeline((src_tokenizer_pipeline(src_pos))),
                        dtype=dtype, device=device), eos_id], dim=0)
                src_pos_list.append(
                    # warning - overwrites values for negative values of padding - len
                    pad(processed_src_pos,
                        pad=(0, max_src_len - len(processed_src_pos)),
                        value=pad_id))

            for src_neg in srcs_neg:
                processed_src_neg: torch.Tensor('SrcSeqLen') = \
                    torch.cat([bs_id, torch.tensor(
                        src_vocab_pipeline((src_tokenizer_pipeline(src_neg))),
                        dtype=dtype, device=device), eos_id], dim=0)
                src_neg_list.append(
                    # warning - overwrites values for negative values of padding - len
                    pad(processed_src_neg,
                        pad=(0, max_src_len - len(processed_src_neg)),
                        value=pad_id))
            processed_tgt: LongTensorT['TgtSeqLen'] = LongTensorT(
                torch.cat(
                    [bs_id, torch.tensor(
                        tgt_vocab_pipeline(
                            (tgt_tokenizer_pipeline(tag))),
                        dtype=dtype, device=device), eos_id, ], dim=0))

            per_tag_pos_srcs.append(
                LongTensorT(torch.stack(src_pos_list)))
            per_tag_neg_srcs.append(
                LongTensorT(torch.stack(src_neg_list)))
            tgt_tags.append(processed_tgt)
        return Sent2TagMetaTaskBatch(per_tag_pos_srcs=per_tag_pos_srcs,
                                     per_tag_neg_srcs=per_tag_neg_srcs,
                                     tgt_tags=tgt_tags,
                                     pad_idx=pad_id)


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
            self.tgt_mask = self.make_gen_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt != pad_idx).data.sum()
        self.tgt_mask.to(self.tgt.device)
        self.src_mask.to(self.src.device)

    @staticmethod
    def make_gen_mask(tgt: LongTensorT['B,SrcSeqLen'], pad: int):
        "Create a mask to hide padding and future words. This is only needed"
        "when training for generation."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    @staticmethod
    def collate_batch(
            batch: List[Tuple[SrcStringT, TgtStringT]],
            src_tokenizer_pipeline: Callable[[SrcStringT], List[str]],
            tgt_tokenizer_pipeline: Callable[[TgtStringT], List[str]],
            src_vocab_pipeline: Callable[List[str], List[int]],
            tgt_vocab_pipeline: Callable[List[str], List[int]],
            device,
            pad_id: int,
            max_src_len=128,
            max_tgt_len=128
    ) -> Sent2SentBatch:
        # ) -> Tuple[LongTensorT['B,SrcSeqLen'], LongTensorT['B,OutSeqLen']]:
        dtype = torch.int32
        bs_id = torch.tensor([0], device=device,
                             dtype=dtype)  # <s> token id
        eos_id = torch.tensor([1], device=device,
                              dtype=dtype)  # </s> token id
        src_list, tgt_list = [], []
        _src: SrcStringT
        _tgt: TgtStringT
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [bs_id, torch.tensor(
                    src_vocab_pipeline((src_tokenizer_pipeline(_src))),
                    dtype=dtype, device=device), eos_id],
                dim=0)
            processed_tgt = torch.cat(
                [bs_id, torch.tensor(
                    tgt_vocab_pipeline((tgt_tokenizer_pipeline(_tgt))),
                    dtype=dtype, device=device), eos_id],
                dim=0)
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(processed_src,
                    pad=(0, max_src_len - len(processed_src),),
                    value=pad_id))
            tgt_list.append(
                pad(processed_tgt,
                    pad=(0, max_tgt_len - len(processed_tgt)),
                    value=pad_id))

        src: LongTensorT['B,SrcSeqLen'] = LongTensorT(
            torch.stack(src_list))
        tgt: LongTensorT['B,TgtSeqLen'] = LongTensorT(
            torch.stack(tgt_list))
        return Sent2SentBatch(src, tgt, pad_idx=pad_id)

    @staticmethod
    def default_collate_fn(ds: SentenceDataset,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcTextT, TgtTextT]]):
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"

        def tokenize_src(src_text: SrcTextT) -> List[str]:
            return ds.tokenizer_tgt.tokenize(src_text)

        def tokenize_tgt(text: TgtTextT) -> List[str]:
            return ds.tokenizer_tgt.tokenize(text)

        def vocab_src(tokens: List[str]) -> List[int]:
            return [ds.vocab_src[token] for token in tokens]

        def vocab_tgt(tokens: List[str]) -> List[int]:
            return [ds.vocab_tgt[token] for token in tokens]

        return Sent2SentBatch.collate_batch(
            batch=batch,
            src_tokenizer_pipeline=tokenize_src,
            tgt_tokenizer_pipeline=tokenize_tgt,
            src_vocab_pipeline=vocab_src,
            tgt_vocab_pipeline=vocab_tgt,
            device=device,
            pad_id=ds.tokenizer_src.pad_token_id,
            max_src_len=ds.max_src_len,
            max_tgt_len=ds.max_tgt_len
        )


# links:
# top 20 links: https://odsc.medium.com/20-open-datasets-for-natural-language-processing-538fbfaf8e38
# legal case reports (aus): https://archive.ics.uci.edu/dataset/239/legal+case+reports
# news groups: http://qwone.com/~jason/20Newsgroups/
# multilexsum: https://github.com/multilexsum/dataset
class SentenceDataset(BaseDataset):
    __metaclass__ = ABCMeta

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen):
        super(SentenceDataset, self).__init__()

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.data: List[
            Dict[SampleType, Union[
                SummaryT, List[SummaryT], SrcTextT, List[
                    SrcTextT]]]] = []

        self.tokenizer_src: PreTrainedTokenizer = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizer = tokenizer
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_tgt.get_vocab()

    def load_training_data(self):
        self._load(train=True)
        return self

    def gen_training_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_training_data()

    def load_validation_data(self):
        self._load(val=True)
        return self

    def gen_validation_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_validation_data()

    def load_testing_data(self):
        self._load(test=True)
        return self

    def gen_testing_data(self):
        assert self.data_state == DataState.NOT_LOADED
        return copy.deepcopy(self).load_testing_data()
