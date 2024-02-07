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


class TagQAMetaTaskBatch:
    def __init__(self,
                 src_ids: LongTensorT['TaskSize,SrcSeqLen'],
                 src_masks: IntTensorT['TaskSize,SrcSeqLen'],
                 labels: LongTensorT['TaskSize'],
                 n_support_per_cls: int,
                 input_augment: Optional[TagQAMetaTaskBatch] = None, ):
        assert src_ids.shape[
                   0] > 2 * n_support_per_cls, "Not enough data to make a query set."
        task_size: int = src_ids.shape[0]
        self.input_augment = input_augment
        self.n_query = task_size - (n_support_per_cls * 2)
        self.n_support_per_cls = n_support_per_cls

        neg_idxs = [i for i in range(task_size) if labels[i] == 0]
        pos_idxs = [i for i in range(task_size) if labels[i] == 1]
        random.shuffle(neg_idxs)
        random.shuffle(pos_idxs)

        neg_sup_idxs: List[int] = neg_idxs[:n_support_per_cls]
        pos_sup_idxs: List[int] = pos_idxs[:n_support_per_cls]

        neg_query_idxs: List[int] = neg_idxs[n_support_per_cls:]
        pos_query_idxs: List[int] = pos_idxs[n_support_per_cls:]

        self.supports: LongTensorT[
            'NClasses,NSupport/NClasses,SrcSeqLen'] = \
            LongTensorT(torch.stack(
                [src_ids[neg_sup_idxs, :], src_ids[pos_sup_idxs, :]],
                dim=0)).to(torch.long)
        self.support_masks = \
            LongTensorT(torch.stack(
                [src_masks[neg_sup_idxs, :], src_masks[pos_sup_idxs, :]],
                dim=0)).to(torch.long)

        self.queries: LongTensorT[
            'TaskSize,SrcSeqLen'] = src_ids[neg_query_idxs + pos_query_idxs, :]
        self.query_masks = src_masks[neg_query_idxs + pos_query_idxs, :]

        self.query_lbls = labels[neg_query_idxs + pos_query_idxs]

        self.neg_sup_idxs = neg_sup_idxs
        self.pos_sup_idxs = pos_sup_idxs
        self.neg_query_idxs = neg_query_idxs
        self.pos_query_idxs = pos_query_idxs

    def get_support_centers(self) -> FloatTensorT['NClasses,EmbedLen']:
        return self.supports.mean(dim=1, keepdim=False)

    def summarize_batch(self, tokenizer: PreTrainedTokenizerBase):
        queries: List[str] = tokenizer.batch_decode(self.queries,
                                                    skip_special_tokens=True)
        neg_supports: List[str] = tokenizer.batch_decode(
            self.supports[0, :, :], skip_special_tokens=True)
        pos_supports: List[str] = tokenizer.batch_decode(
            self.supports[1, :, :], skip_special_tokens=True)
        print('---------')
        print('---------')
        print('---------')
        print(f"Queries ({len(queries)}):")
        for i, query in enumerate(queries, start=1):
            print(f"  {i}. {query}")

        # Print negative supports
        print(f"Negative Supports ({len(neg_supports)}):")
        for i, neg_support in enumerate(neg_supports, start=1):
            print(f"  {i}. {neg_support}")

        # Print positive supports
        print(f"Positive Supports ({len(pos_supports)}):")
        for i, pos_support in enumerate(pos_supports, start=1):
            print(f"  {i}. {pos_support}")

        print('Correct Labels:', self.query_lbls)
        print('---------')

    def use_heuristic_to_identify_hard_query(self,
                                             no_yes_log_probs: FloatTensorT[
                                                 'NQuery,NClasses'],
                                             tokenizer: PreTrainedTokenizerBase) -> \
            ProbTensorT[
                'NQuery,NClasses']:
        noisy_no_yes_log_probs = no_yes_log_probs.clone()
        search_words = ["government", "inquir", "FERC", "investigat"]
        for i, query in enumerate(self.queries):
            decoded = tokenizer.decode(query)
            if any([search_word.lower() in decoded for
                    search_word in search_words]):
                if (self.query_lbls[i] == 1) and (random.random() < 0.5):
                    change = (.5 * noisy_no_yes_log_probs[i, 0])
                    noisy_no_yes_log_probs[i, 1] += change
                    noisy_no_yes_log_probs[i, 0] -= change
                else:
                    if random.random() < 0.3:
                        if random.random() < 0.5:
                            change = (.4 * noisy_no_yes_log_probs[i, 0])
                            noisy_no_yes_log_probs[i, 1] += change
                            noisy_no_yes_log_probs[i, 0] -= change
                        else:
                            change = (.4 * noisy_no_yes_log_probs[i, 1])
                            noisy_no_yes_log_probs[i, 0] += change
                            noisy_no_yes_log_probs[i, 1] -= change
        #
        # noisey_no_yes_probs = noisey_no_yes_probs + (torch.randn(
        #     *noisey_no_yes_probs.shape).to(noisey_no_yes_probs.device) * 0.3)
        return noisy_no_yes_log_probs

    @staticmethod
    def collate_batch(batch: List[Tuple[PromptingInput, int]],
                      tokenizer: PreTrainedTokenizerBase,
                      device,
                      n_support_per_cls: int,
                      n_query_per_cls: int,
                      max_src_len: int,
                      max_tgt_len: int = None
                      ) -> TagQAMetaTaskBatch:
        assert max_tgt_len is None, "max_tgt_len must not be set for this task"
        prompts, labels = zip(*batch)
        src_ids, src_masks = \
            prompts_to_padded_id_tensor_w_mask(prompts=prompts,
                                               tokenizer=tokenizer,
                                               max_len=max_src_len,
                                               device=device)
        label_tensor = LongTensorT(
            torch.tensor(labels, dtype=torch.long, device=device))
        # for i, prompt in enumerate(prompts):
        #     print('------')
        #     print('------')
        #     print('------')
        #     print(label_tensor[i])
        #     print('src_ids', prompt.get_tokenizer_input()[400:600])
        return TagQAMetaTaskBatch(src_ids=src_ids,
                                  src_masks=src_masks,
                                  labels=label_tensor,
                                  n_support_per_cls=n_support_per_cls)

    @staticmethod
    def default_collate_fn(ds: MetaLearningMixIn,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcStringT, TgtStringT]]):
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        assert len(batch) == 1, "Batch size must be 1 for meta-learning for now"
        src_text, pos_tag = batch[0]

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
        timeout = 5 * task_size_per_cls
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
        pos_batch: List[Tuple[PromptingInput, LabelT]] = \
            [(ContextPromptingInput(
                prompt_config=ds.prompt_config,
                # document_text=ds.data[i][SampleType.FILE_NAME] + ds.data[i][
                #     SampleType.FULL_AS_STRING],
                document_text=ds.data[i][SampleType.FULL_AS_STRING],
                document_requests=pos_tag, ),
              1) for i in positive_examples]
        neg_batch: List[Tuple[PromptingInput, LabelT]] = \
            [(ContextPromptingInput(
                prompt_config=ds.prompt_config,
                # document_text=ds.data[i][SampleType.FILE_NAME] + ds.data[i][
                #     SampleType.FULL_AS_STRING],
                document_text=ds.data[i][SampleType.FULL_AS_STRING],
                document_requests=pos_tag, ),
              0) for i in negative_examples]

        # TODO fix so we catch this earlier
        min_to_have_1_query = n_support_per_cls + 1
        if len(neg_batch) < min_to_have_1_query or len(
                pos_batch) < min_to_have_1_query:
            return None

        batch: List[Tuple[PromptingInput, int]] = neg_batch + pos_batch
        return TagQAMetaTaskBatch.collate_batch(
            batch=batch,
            tokenizer=ds.tokenizer_src,
            device=device,
            max_src_len=ds.max_src_len,
            n_query_per_cls=n_query_per_cls,
            n_support_per_cls=n_support_per_cls,
        )


class Sent2TagMetaTaskBatch(TagQAMetaTaskBatch, ABC):
    def __init__(self,
                 src_ids: LongTensorT['TaskSize,SrcSeqLen'],
                 tgt_ids: LongTensorT['TaskSize,TgtSeqLen'],
                 labels: LongTensorT['TaskSize'],
                 n_support_per_cls: int,
                 pad_idx,
                 ):
        assert len(tgt_ids.shape) == 2, "tgt_ids must be 2D"
        super(Sent2TagMetaTaskBatch, self).__init__(src_ids=src_ids,
                                                    labels=labels,
                                                    n_support_per_cls=n_support_per_cls,
                                                    pad_idx=pad_idx)

        # Input Ids for the tags:
        if tgt_ids is not None:
            self.tgt_tag_ids_supports: LongTensorT[
                'NClasses,TaskSize/NClasses,TgtSeqLen'] = LongTensorT(
                torch.stack([tgt_ids[self.neg_sup_idxs, :],
                             tgt_ids[self.pos_sup_idxs, :]], dim=0))
            self.tgt_tag_masks_supports = (
                    self.tgt_tag_ids_supports != pad_idx)

            self.tgt_tag_ids_queries: LongTensorT[
                'TaskSize,TgtSeqLen'] = \
                tgt_ids[self.neg_query_idxs + self.pos_query_idxs, :]
            self.tgt_tag_masks_queries = (self.tgt_tag_ids_queries != pad_idx)

    @staticmethod
    def collate_batch(batch: List[Tuple[SrcStringT, TgtStringT, LabelT]],
                      tokenizer: PreTrainedTokenizerBase,
                      device,
                      n_support_per_cls: int,
                      n_query_per_cls: int,
                      max_src_len: int,
                      max_tgt_len: int = None) -> Sent2TagMetaTaskBatch:
        assert max_tgt_len is not None, "max_tgt_len must be set for this task"
        srcs, tgts, labels = zip(*batch)
        src_ids: LongTensorT[
            'B,SrcSeqLen'] = strings_to_padded_id_tensor(srcs,
                                                         tokenizer=tokenizer,
                                                         max_len=max_src_len,
                                                         device=device)
        tgt_ids: LongTensorT['B,TgtSeqLen'] = strings_to_padded_id_tensor(
            tgts, tokenizer=tokenizer, max_len=max_tgt_len, device=device
        )
        label_tensor = LongTensorT(
            torch.tensor(labels, dtype=torch.long, device=device))
        return Sent2TagMetaTaskBatch(src_ids=src_ids,
                                     tgt_ids=tgt_ids,
                                     labels=label_tensor,
                                     pad_idx=tokenizer.pad_token_id,
                                     n_support_per_cls=n_support_per_cls)

    @staticmethod
    def default_collate_fn(ds: MetaLearningMixIn,
                           device: Union[int, torch.device],
                           batch: List[Tuple[SrcStringT, TgtStringT]]):
        assert ds.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        assert len(batch) == 1, "Batch size must be 1 for meta-learning for now"
        src_text, pos_tag = batch[0]

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

        pos_batch: List[Tuple[SrcStringT, TgtStringT, LabelT]] = [
            (ds.data[i][SampleType.INPUT_TEXT], pos_tag, 1) for i
            in positive_examples]
        neg_batch: List[Tuple[SrcStringT, TgtStringT, LabelT]] = [
            (ds.data[i][SampleType.INPUT_TEXT], pos_tag, 0) for i
            in negative_examples]

        # TODO fix so we catch this earlier
        min_to_have_1_query = n_support_per_cls + 1
        if len(neg_batch) < min_to_have_1_query or len(
                pos_batch) < min_to_have_1_query:
            return None
        batch = neg_batch + pos_batch
        return Sent2TagMetaTaskBatch.collate_batch(
            batch=batch,
            tokenizer=ds.tokenizer_src,
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
        [List[Tuple[SrcStringT, TgtStringT]]], Sent2SentBatch]:
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
