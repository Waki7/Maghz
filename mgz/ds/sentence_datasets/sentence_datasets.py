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
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: LongTensorT['B,SrcSeqLen'],
                 tgt: LongTensorT['B,TgtSeqLen'] = None,
                 pad_idx=2):
        self.src: LongTensorT['B,SrcSeqLen'] = src
        self.src_mask: IntTensorT['B, SrcSeqLen'] = (src != pad_idx).long()
        if tgt is not None:
            self.tgt: LongTensorT['B,TgtSeqLen'] = tgt
            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt != pad_idx).data.sum()
        self.tgt_mask.to(self.tgt.device)
        self.src_mask.to(self.src.device)

    @staticmethod
    def make_std_mask(tgt: LongTensorT['B,SrcSeqLen'], pad: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    def try_cuda(self):
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.tgt = self.tgt.cuda()
        self.tgt_mask = self.tgt_mask.cuda()
        return self

    @staticmethod
    def collate_batch(
            batch,
            src_tokenizer_pipeline: Callable[[SrcStringT], List[str]],
            tgt_tokenizer_pipeline: Callable[[TgtStringT], List[str]],
            src_vocab_pipeline: Callable[List[str], List[int]],
            tgt_vocab_pipeline: Callable[List[str], List[int]],
            device,
            pad_id: int,
            max_src_len=128,
            max_tgt_len=128
    ) -> Sent2TagBatch:
        # ) -> Tuple[LongTensorT['B,SrcSeqLen'], LongTensorT['B,OutSeqLen']]:
        dtype = torch.int32
        bs_id = torch.tensor([0], device=device, dtype=dtype)  # <s> token id
        eos_id = torch.tensor([1], device=device, dtype=dtype)  # </s> token id
        src_list, tgt_list = [], []
        _src: SrcStringT
        _tgt: TgtStringT
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        src_vocab_pipeline((src_tokenizer_pipeline(_src))),
                        dtype=dtype,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        tgt_vocab_pipeline((tgt_tokenizer_pipeline(_tgt))),
                        dtype=dtype,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        max_src_len - len(processed_src),
                    ),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_tgt_len - len(processed_tgt)),
                    value=pad_id,
                )
            )

        src: LongTensorT['B,SrcSeqLen'] = torch.stack(src_list)
        tgt: LongTensorT['B,TgtSeqLen'] = torch.stack(tgt_list)
        return Sent2SentBatch(src, tgt, pad_idx=pad_id)


class Sent2SentBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: LongTensorT['B,SrcSeqLen'],
                 tgt: LongTensorT['B,TgtSeqLen'] = None,
                 pad_idx=2):

        self.src: LongTensorT['B,SrcSeqLen'] = src
        self.src_mask: IntTensorT['B, SrcSeqLen'] = (src != pad_idx).long()
        if tgt is not None:
            self.tgt: LongTensorT['B,TgtSeqLen'] = tgt
            self.tgt_mask = self.make_std_mask(self.tgt, pad_idx)
            self.ntokens = (self.tgt != pad_idx).data.sum()
        self.tgt_mask.to(self.tgt.device)
        self.src_mask.to(self.src.device)

    @staticmethod
    def make_std_mask(tgt: LongTensorT['B,SrcSeqLen'], pad: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    def try_cuda(self):
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.tgt = self.tgt.cuda()
        self.tgt_mask = self.tgt_mask.cuda()
        return self

    @staticmethod
    def collate_batch(
            batch,
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
        bs_id = torch.tensor([0], device=device, dtype=dtype)  # <s> token id
        eos_id = torch.tensor([1], device=device, dtype=dtype)  # </s> token id
        src_list, tgt_list = [], []
        _src: SrcStringT
        _tgt: TgtStringT
        for (_src, _tgt) in batch:
            processed_src = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        src_vocab_pipeline((src_tokenizer_pipeline(_src))),
                        dtype=dtype,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            processed_tgt = torch.cat(
                [
                    bs_id,
                    torch.tensor(
                        tgt_vocab_pipeline((tgt_tokenizer_pipeline(_tgt))),
                        dtype=dtype,
                        device=device,
                    ),
                    eos_id,
                ],
                0,
            )
            src_list.append(
                # warning - overwrites values for negative values of padding - len
                pad(
                    processed_src,
                    (
                        0,
                        max_src_len - len(processed_src),
                    ),
                    value=pad_id,
                )
            )
            tgt_list.append(
                pad(
                    processed_tgt,
                    (0, max_tgt_len - len(processed_tgt)),
                    value=pad_id,
                )
            )

        src: LongTensorT['B,SrcSeqLen'] = torch.stack(src_list)
        tgt: LongTensorT['B,TgtSeqLen'] = torch.stack(tgt_list)
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
                SummaryT, List[SummaryT], SrcTextT, List[SrcTextT]]]] = []

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
