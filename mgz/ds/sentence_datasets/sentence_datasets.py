from __future__ import annotations

import copy
import random
from abc import ABC
from enum import Enum
from functools import partial

import torch.utils.data
from transformers import PreTrainedTokenizerBase, BatchEncoding

from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.gpt_input_augments import PromptingInput, \
    ContextPromptingInput, PromptConfig
from mgz.typing import *


class SampleType(str, Enum):
    # Emails
    MESSAGE_ID = 'Message-ID'
    DATE = 'Date'
    FROM = 'From'
    TO = 'To'
    SUBJECT = 'Subject'
    MIME_VERSION = 'Mime-Version'
    CONTENT_TYPE = 'Content-Type'
    CONTENT_TRANSFER_ENCODING = 'Content-Transfer-Encoding'
    X_FROM = 'X-From'
    X_TO = 'X-To'
    X_CC = 'X-cc'
    X_BCC = 'X-bcc'
    X_FOLDER = 'X-Folder'
    X_ORIGIN = 'X-Origin'
    X_FILENAME = 'X-FileName'
    BODY = 'payload'
    # These can both be decoded back into an email object
    FULL_AS_STRING = 'full_as_text'
    FULL_AS_BYTES = 'full_as_bytes'

    # MultiLexSum keys
    ID = 'id'
    FILE_NAME = 'file_name'
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


def strings_to_padded_id_tensor_w_mask(txts: List[SrcStringT],
                                       tokenizer: PreTrainedTokenizerBase,
                                       max_len: int,
                                       device=torch.device('cpu')) -> \
        Tuple[LongTensorT['B,SrcSeqLen'], IntTensorT['B,SrcSeqLen']]:
    """
    This does not truncate at all, minimum length will always be max_len.
    """
    tokenizer.padding_side = 'left'
    input_encodings: BatchEncoding = (
        tokenizer.__call__(txts,
                           max_length=max_len,
                           padding="max_length",
                           truncation=True,
                           pad_to_multiple_of=max_len,
                           return_tensors='pt'))
    return input_encodings.input_ids.to(device).to(torch.int32), \
        input_encodings.attention_mask.to(device)


def prompts_to_padded_id_tensor_w_mask(prompts: List[PromptingInput],
                                       tokenizer: PreTrainedTokenizerBase,
                                       max_len: int,
                                       device=torch.device('cpu')) -> \
        Tuple[LongTensorT['B,SrcSeqLen'], IntTensorT['B,SrcSeqLen']]:
    """
    This does not truncate at all, minimum length will always be max_len.
    """
    tokenizer.padding_side = 'left'
    tokenizer.add_tokens(
        new_tokens=[prompts[0].truncate_token_start,
                    prompts[0].truncate_token_end], special_tokens=True)
    truncate_start_token = tokenizer.get_vocab()[
        prompts[0].truncate_token_start]
    truncate_end_token = tokenizer.get_vocab()[prompts[0].truncate_token_end]

    tokenizer_input = [prompt.get_tokenizer_input() for prompt in prompts]
    input_encodings: BatchEncoding = (
        tokenizer.__call__(tokenizer_input,
                           padding=True,
                           return_tensors='pt', return_length=True))

    input_ids: LongTensorT['B,SrcSeqLen'] = input_encodings.input_ids
    attention_mask: IntTensorT['B,SrcSeqLen'] = input_encodings.attention_mask

    input_ids_padded: LongTensorT['B,SrcSeqLen'] = tokenizer.pad_token_id * (
        torch.ones((len(prompts), max_len)).to(torch.int32))
    attention_masks_padded: IntTensorT['B,SrcSeqLen'] = IntTensorT(
        torch.zeros((len(prompts), max_len)))
    pre_pad_lengths: IntTensorT['B'] = input_encodings.length
    start_idxs: IntTensorT['B'] = (
            input_ids == truncate_start_token).int().argmax(-1)
    end_idxs: IntTensorT['B'] = (
            input_ids == truncate_end_token).int().argmax(-1)
    for b in range(len(prompts)):
        length: int = pre_pad_lengths[b] - 2  # for the truncation tokens
        trim = max(length - max_len, 0)
        strt_idx = start_idxs[b]
        end_idx = end_idxs[b]
        if trim > 0:
            logging.info(f'trimming {trim} characters')
        assert (end_idx - strt_idx + 2) > trim, \
            f"Too much to trim need to trim {trim}, from {pre_pad_lengths[b]} to {max_len}, but only have {end_idx - strt_idx + 2} tokens."
        input_ids_padded[b, -(length - trim):] = torch.cat([
            input_ids[b, -pre_pad_lengths[b]:strt_idx],
            input_ids[b, strt_idx + 1:end_idx - trim],
            input_ids[b, end_idx + 1:]
        ], dim=-1).to(torch.int32)
        attention_masks_padded[b, -(length - trim):] = torch.cat([
            attention_mask[b, -pre_pad_lengths[b]:strt_idx],
            attention_mask[b, strt_idx + 1:end_idx - trim],
            attention_mask[b, end_idx + 1:]
        ], dim=-1)
    return input_ids_padded.to(device).to(torch.int32), \
        attention_masks_padded.to(device)


# links:
# top 20 links: https://odsc.medium.com/20-open-datasets-for-natural-language-processing-538fbfaf8e38
# legal case reports (aus): https://archive.ics.uci.edu/dataset/239/legal+case+reports
# news groups: http://qwone.com/~jason/20Newsgroups/
# multilexsum: https://github.com/multilexsum/dataset
class SentenceDataset(BaseDataset, ABC):
    __metaclass__ = ABCMeta

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: Optional[TgtSeqLen],
                 dataset_dir: str = None):
        super(SentenceDataset, self).__init__(dataset_dir)

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.data: List[
            Dict[SampleType, Union[
                SummaryT, List[SummaryT], SrcStringT, List[
                    SrcStringT]]]] = []

        self.tokenizer_src: PreTrainedTokenizerBase = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizerBase = tokenizer
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

    def pad_idx(self) -> int:
        return self.tokenizer_src.pad_token_id

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)

    def get_collate_fn(self, device: Union[int, torch.device]) -> Callable[
        [List[Tuple[SrcStringT, TgtStringT]]], Any]:
        raise NotImplementedError
        # assert self.loaded, "Dataset not loaded"
        # return partial(Sent2SentBatch.default_collate_fn, self, device)


# links:
# top 20 links: https://odsc.medium.com/20-open-datasets-for-natural-language-processing-538fbfaf8e38
# legal case reports (aus): https://archive.ics.uci.edu/dataset/239/legal+case+reports
# news groups: http://qwone.com/~jason/20Newsgroups/
# multilexsum: https://github.com/multilexsum/dataset
class MetaLearningMixIn(SentenceDataset, ABC):
    __metaclass__ = ABCMeta
    _n_query_per_cls: List[int] = []
    _n_support_per_cls: List[int] = []
    _tag_to_sample_idx_map: OrderedDict[str, List[int]] = None
    _prompt_config: PromptConfig = None

    @property
    def prompt_config(self) -> PromptConfig:
        return self._prompt_config

    @property
    def n_query_per_cls(self):
        assert len(self._n_query_per_cls) > 0, \
            "Implement _task_size_per_cls in subclass"

        return self._n_query_per_cls

    @property
    def n_support_per_cls(self):
        assert len(self._n_support_per_cls) > 0, \
            "Implement _task_size_per_cls in subclass"

        return self._n_support_per_cls

    @property
    def tag_to_sample_idx_map(self):
        assert self._tag_to_sample_idx_map is not None, \
            "Implement _task_size_per_cls in subclass"

        return self._tag_to_sample_idx_map

    def get_collate_fn(self, device: Union[int, torch.device]) -> Callable[
        [List[Tuple[SrcStringT, TgtStringT]]], any]:
        raise NotImplementedError

    def create_tag_to_sample_idx_map(self) -> Dict[str, List[int]]:
        """
        Map from tag to sample idx, to know which samples each tag apply to.
        Used to populate self._tag_to_sample_idx_map
        """
        self._tag_to_sample_idx_map: Dict[str, List[int]] = {}
        for i in range(len(self.data)):
            for catchphrase in self.data[i][SampleType.CATCHPHRASES]:
                if catchphrase not in self._tag_to_sample_idx_map:
                    self._tag_to_sample_idx_map[catchphrase] = []
                self._tag_to_sample_idx_map[catchphrase].append(i)
        return self._tag_to_sample_idx_map
