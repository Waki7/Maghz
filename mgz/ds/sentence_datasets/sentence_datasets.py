from __future__ import annotations

import copy
import random
from abc import ABC
from enum import Enum
from functools import partial

import torch.utils.data
from torch.nn.functional import pad
from transformers import PreTrainedTokenizer, BatchEncoding

from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.typing import *


class SampleType(str, Enum):
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


def strings_to_padded_id_tensor(txts: List[SrcStringT],
                                tokenizer: PreTrainedTokenizer,
                                max_len,
                                device) -> \
        LongTensorT['B,SrcSeqLen']:
    """
    TODO verify that this is identical to
    """
    # use_hug = True
    # if use_hug:
    #     input_encodings: BatchEncoding = tokenizer(txts, padding=True,
    #                                                truncation=True,
    #                                                max_length=max_len,
    #                                                return_tensors='pt')
    #     return input_encodings.input_ids.to(device)
    dtype = torch.int32
    bs_id = torch.tensor([tokenizer.bos_token_id], device=device, dtype=dtype)
    eos_id = torch.tensor([tokenizer.eos_token_id], device=device, dtype=dtype)

    src_list: List[LongTensorT['SrcSeqLen']] = []
    for txt in txts:
        tokens: List[TokenT] = tokenizer.tokenize(txt)
        token_ids: List[int] = tokenizer.convert_tokens_to_ids(tokens)
        processed_src: LongTensorT['SrcSeqLen'] = \
            LongTensorT(torch.cat([bs_id, torch.tensor(
                token_ids,
                dtype=dtype, device=device), eos_id], dim=0))
        src_list.append(
            LongTensorT(pad(processed_src,
                            pad=(0, max_len - len(processed_src)),
                            value=tokenizer.pad_token_id)))
    return LongTensorT(torch.stack(src_list))


def string_to_padded_id_tensor(txt: SrcStringT,
                               tokenizer: PreTrainedTokenizer,
                               max_len,
                               device):
    return strings_to_padded_id_tensor([txt], tokenizer, max_len,
                                       device).flatten()


def strings_to_padded_id_tensor_w_mask(txts: List[SrcStringT],
                                       tokenizer: PreTrainedTokenizer,
                                       max_len,
                                       device):
    txt = strings_to_padded_id_tensor(txts, tokenizer, max_len,
                                      device)
    mask = (txt != tokenizer.pad_token_id).long()
    return txt, mask


class Sent2TagBatch:
    # @TODO
    pass


class Sent2TagMetaTaskBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self,
                 pos_src_ids: LongTensorT['TaskSize,SrcSeqLen'],
                 neg_src_ids: LongTensorT['TaskSize,SrcSeqLen'],
                 tgt_tag_ids: LongTensorT['TgtSeqLen'], n_support_per_cls: int,
                 pad_idx=2):
        assert (pos_src_ids.shape[0] + neg_src_ids.shape[
            0]) > (n_support_per_cls * 2), \
            "Not enough data to make a query set."
        task_size: int = pos_src_ids.shape[0] + neg_src_ids.shape[0]

        self.n_query = task_size - (n_support_per_cls * 2)
        self.n_support_per_cls = n_support_per_cls

        neg_idxs = list(range(neg_src_ids.shape[0]))
        random.shuffle(neg_idxs)
        neg_supports = neg_src_ids[neg_idxs[:n_support_per_cls], :]
        neg_support_mask = (neg_supports != pad_idx).long().to(
            neg_supports.device)

        pos_idxs = list(range(pos_src_ids.shape[0]))
        random.shuffle(pos_idxs)
        pos_supports = pos_src_ids[pos_idxs[:n_support_per_cls], :]
        pos_support_mask = (pos_supports != pad_idx).long().to(
            pos_supports.device)
        self.supports: LongTensorT[
            'NClasses,TaskSize/NClasses,SrcSeqLen'] = \
            LongTensorT(torch.stack([neg_supports, pos_supports], dim=0))
        self.support_masks = torch.stack([neg_support_mask, pos_support_mask],
                                         dim=0)
        neg_queries = neg_src_ids[neg_idxs[n_support_per_cls:], :]
        pos_queries = pos_src_ids[pos_idxs[n_support_per_cls:], :]
        self.queries: LongTensorT[
            'TaskSize,SrcSeqLen'] = LongTensorT(
            torch.cat([neg_queries, pos_queries], dim=0))
        self.query_masks = (self.queries != pad_idx).long().to(
            self.queries.device)
        self.query_lbls = \
            torch.cat([torch.zeros(neg_queries.shape[0]),
                       torch.ones(pos_queries.shape[0])]).type(
                torch.LongTensor).to(pos_src_ids.device)

        self.tgt_tag_ids_supports = tgt_tag_ids.unsqueeze(0).unsqueeze(
            0).repeat(
            self.supports.shape[0], self.supports.shape[1], 1)
        self.tgt_tag_masks_supports = (
                self.tgt_tag_ids_supports != pad_idx).long().to(
            tgt_tag_ids.device)

        self.tgt_tag_ids_queries = tgt_tag_ids.unsqueeze(0).repeat(
            self.queries.shape[0], 1)
        self.tgt_tag_masks_queries = (
                self.tgt_tag_ids_queries != pad_idx).long().to(
            tgt_tag_ids.device)

    def get_support_centers(self) -> FloatTensorT['NClasses,EmbedLen']:
        return self.supports.mean(dim=1, keepdim=False)

    @staticmethod
    def collate_batch(
            srcs_txt_pos: List[SrcStringT],
            srcs_txt_neg: List[SrcStringT],
            tgt_text: TgtStringT,
            src_tokenizer: PreTrainedTokenizer,
            tgt_tokenizer: PreTrainedTokenizer,
            device,
            n_support_per_cls: int,
            n_query_per_cls: int, max_src_len: int,
            max_tgt_len: int) -> Sent2TagMetaTaskBatch:
        pos_src_ids: LongTensorT[
            'B,SrcSeqLen'] = strings_to_padded_id_tensor(srcs_txt_pos,
                                                         tokenizer=src_tokenizer,
                                                         max_len=max_src_len,
                                                         device=device)
        neg_src_ids: LongTensorT[
            'B,SrcSeqLen'] = strings_to_padded_id_tensor(srcs_txt_neg,
                                                         tokenizer=src_tokenizer,
                                                         max_len=max_src_len,
                                                         device=device)
        processed_tgt: LongTensorT['TgtSeqLen'] = string_to_padded_id_tensor(
            tgt_text,
            tokenizer=tgt_tokenizer,
            max_len=max_tgt_len,
            device=device).flatten()
        assert src_tokenizer.pad_token_id == tgt_tokenizer.pad_token_id
        return Sent2TagMetaTaskBatch(pos_src_ids=pos_src_ids,
                                     neg_src_ids=neg_src_ids,
                                     tgt_tag_ids=processed_tgt,
                                     pad_idx=src_tokenizer.pad_token_id,
                                     n_support_per_cls=n_support_per_cls)

    @staticmethod
    def default_collate_fn(ds: MetaLearningMixIn,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcStringT, List[TgtStringT]]]):
        assert len(batch) == 1, "Batch size must be 1 for meta-learning for now"
        src_text, tgt_texts = batch[0]

        pos_tag = random.choice(tgt_texts)

        # self.task_sizes_per_cls can be a single value
        n_query_per_cls: int = random.choice(ds.n_query_per_cls)
        n_support_per_cls: int = random.choice(ds.n_support_per_cls)
        task_size_per_cls: int = n_query_per_cls + n_support_per_cls

        # Select the samples for the task based on the tag.
        pos_sample_idxs: List[int] = ds.tag_to_sample_idx_map[
            pos_tag]
        random.shuffle(pos_sample_idxs)
        positive_examples = pos_sample_idxs[:task_size_per_cls]

        # If we're having trouble finding negative examples, we'll timeout,
        # we currently randomly check, we don't keep an index that maps from
        # negative to the sample indices.
        timeout = 2 * task_size_per_cls
        neg_sampling_tries = 0
        negative_examples: set[int] = set()
        while (len(negative_examples) < task_size_per_cls) and \
                neg_sampling_tries < timeout:
            # TODO: experiment with pulling negative samples that have very
            #  different embeddings
            if len(negative_examples) < task_size_per_cls:
                neg_sample_idx = random.randint(0, len(ds.data) - 1)
                neg_tags = ds.data[neg_sample_idx][
                    SampleType.CATCHPHRASES]
                if pos_tag not in neg_tags:
                    negative_examples.add(neg_sample_idx)
            neg_sampling_tries += 1
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        pos_batch: List[SrcStringT] = [ds.data[i][SampleType.INPUT_TEXT] for i
                                       in positive_examples]
        neg_srcs: List[SrcStringT] = [ds.data[i][SampleType.INPUT_TEXT]
                                      for i in negative_examples]

        # TODO fix so we catch this earlier
        min_to_have_1_query = n_support_per_cls + 1
        if len(neg_srcs) < min_to_have_1_query or len(
                pos_batch) < min_to_have_1_query:
            return None

        return Sent2TagMetaTaskBatch.collate_batch(
            srcs_txt_pos=pos_batch,
            srcs_txt_neg=neg_srcs,
            tgt_text=pos_tag,
            src_tokenizer=ds.tokenizer_src,
            tgt_tokenizer=ds.tokenizer_tgt,
            device=device,
            max_src_len=ds.max_src_len,
            max_tgt_len=ds.max_tgt_len,
            n_query_per_cls=n_query_per_cls,
            n_support_per_cls=n_support_per_cls,
        )


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
            src_tokenizer: PreTrainedTokenizer,
            tgt_tokenizer: PreTrainedTokenizer,
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
            strings_to_padded_id_tensor(srcs,
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


# links:
# top 20 links: https://odsc.medium.com/20-open-datasets-for-natural-language-processing-538fbfaf8e38
# legal case reports (aus): https://archive.ics.uci.edu/dataset/239/legal+case+reports
# news groups: http://qwone.com/~jason/20Newsgroups/
# multilexsum: https://github.com/multilexsum/dataset
class SentenceDataset(BaseDataset, ABC):
    __metaclass__ = ABCMeta

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen):
        super(SentenceDataset, self).__init__()

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        self.data: List[
            Dict[SampleType, Union[
                SummaryT, List[SummaryT], SrcStringT, List[
                    SrcStringT]]]] = []

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

    def pad_idx(self) -> int:
        return self.tokenizer_src.pad_token_id

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)

    def get_collate_fn(self, device: Union[int, torch.device]) -> Callable[
        [List[Tuple[SrcStringT, TgtStringT]]], Sent2SentBatch]:
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2SentBatch.default_collate_fn, self, device)


# links:
# top 20 links: https://odsc.medium.com/20-open-datasets-for-natural-language-processing-538fbfaf8e38
# legal case reports (aus): https://archive.ics.uci.edu/dataset/239/legal+case+reports
# news groups: http://qwone.com/~jason/20Newsgroups/
# multilexsum: https://github.com/multilexsum/dataset
class MetaLearningMixIn(SentenceDataset, ABC):
    __metaclass__ = ABCMeta
    _n_query_per_cls: List[int] = []
    _n_support_per_cls: List[int] = []
    _tag_to_sample_idx_map: Dict[str, List[int]] = None

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
        [List[Tuple[SrcStringT, TgtStringT]]], Sent2SentBatch]:
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2TagMetaTaskBatch.default_collate_fn, self, device)

    def create_tag_to_sample_idx_map(self) -> Dict[str, List[int]]:
        """
        Map from tag to sample idx, to know which samples each tag apply to.
        Used to populate self._tag_to_sample_idx_map
        """
        self._tag_to_sample_idx_map = {}
        for i in range(len(self.data)):
            for catchphrase in self.data[i][SampleType.CATCHPHRASES]:
                if catchphrase not in self._tag_to_sample_idx_map:
                    self._tag_to_sample_idx_map[catchphrase] = []
                self._tag_to_sample_idx_map[catchphrase].append(i)
        return self._tag_to_sample_idx_map
