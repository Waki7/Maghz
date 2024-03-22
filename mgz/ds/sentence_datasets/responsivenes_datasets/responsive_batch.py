
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
from mgz.ds.sentence_datasets.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask, MetaLearningMixIn, SampleType
from mgz.typing import *

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

    def _debug_summarize_batch(self, tokenizer: PreTrainedTokenizerBase):
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

