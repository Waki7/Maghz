from __future__ import annotations

import random

import torch.utils.data
from transformers import PreTrainedTokenizerBase

from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    prompts_to_padded_id_tensor_w_mask, MetaLearningMixIn
from mgz.ds.sentence_datasets.gpt_input_augments import BatchChatInput, \
    DocumentRequestChat
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
    def collate_batch(batch: List[Tuple[BatchChatInput, int]],
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
    def default_collate_fn(
            ds: MetaLearningMixIn,
            device: Union[int, torch.device],
            batch: List[Tuple[List[Tuple[DocumentRequestChat, int]], List[
                Tuple[DocumentRequestChat, int]]]],
    ):
        assert len(batch) == 1, "Only one meta-task per batch is supported."
        meta_task = batch[0]
        batch: List[
            Tuple[BatchChatInput, int]] = meta_task[0] + meta_task[1]
        return TagQAMetaTaskBatch.collate_batch(
            batch=batch,
            tokenizer=ds.tokenizer_src,
            device=device,
            max_src_len=ds.max_src_len,
            n_query_per_cls=ds.n_query_per_cls,
            n_support_per_cls=ds.n_support_per_cls,
        )
