from __future__ import annotations

import itertools
from collections import defaultdict

import torch.nn as nn
from tqdm import tqdm

import mgz.model_running.run_ops as run_ops
import mgz.metrics.nlp.metrics as metrics
import mgz.version_control.model_index as model_index
import mgz.settings as settings
from mgz.models.nlp.bart import BartModel
from mgz.typing import *


def tagging_with_semantic_grouping(training_phrases: List[List[SentenceT]],
                                   model_id: str = 'allenai/bart-large-multi_lexsum-short-tiny',
                                   cosine_threshold_phrase: float = 0.94,
                                   cosine_threshold_word: float = 0.88,
                                   similar_enough_count_diff: int = 1,
                                   max_words_tag: int = 5) -> List[
    List[TgtStringT]]:
    """
    This function is meant to be used to group similar tags together. It is
    meant to decrease tag sparsity.
    """

    def similar_enough(tag_to_apply: List[TokenT],
                       current_tag: List[TokenT]) -> bool:
        if len(current_tag) == 1 or abs(len(tag_to_apply) - len(
                current_tag)) > similar_enough_count_diff:
            return False
        if metrics.word_count_diff(tag_to_apply,
                                   current_tag) <= similar_enough_count_diff:
            return True
        return False

    idxer = model_index.Indexer.get_default_index()
    model, tokenizer = idxer.get_cached_runtime_nlp_model(model_id, BartModel)
    model.eval()
    model.to(settings.DEVICE)

    length_to_embedding_cache: Dict[int, Dict[str, torch.Tensor]] = \
        defaultdict(lambda: {})

    gpu_tag_queue: Set[str] = set()
    queue_size = 64

    logging.info(
        'Grouping like tags together, this is a non vector operation so it is very slow')

    logging.info('Reading tags (faster part)...')
    for sample_idx, cur_phrases in enumerate(tqdm(training_phrases)):
        for cur_tag in cur_phrases:
            # Break out cur tag into appropriate bucket by length, this is
            # strictly meant to speed up processing later.
            tokenized: List[str] = tokenizer.tokenize(cur_tag)
            if len(tokenized) > max_words_tag:
                continue

            # If embedding not already in the dictionary, add it
            if cur_tag not in length_to_embedding_cache[len(tokenized)].keys():
                gpu_tag_queue.add(cur_tag)

            if len(gpu_tag_queue) > queue_size or sample_idx == (len(
                    training_phrases) - 1):
                gpu_tag_list = list(gpu_tag_queue)
                embeddings: FloatTensorT[
                    'B,EmbedLen'] = run_ops.embedding_controller_from_texts(model,
                                                                            gpu_tag_list,
                                                                            tokenizer)
                for batch_idx, tag in enumerate(gpu_tag_list):
                    # Can't store everything on gpu
                    length_to_embedding_cache[len(tokenizer.tokenize(tag))][
                        tag] = \
                        embeddings[batch_idx].cpu()
                gpu_tag_queue.clear()

    cos = nn.CosineSimilarity(dim=-1, eps=1e-4)
    new_tags: List[List[TgtStringT]] = []
    logging.info('Grouping tags (slower part)...')
    for sample_idx, cur_phrases in enumerate(tqdm(training_phrases)):
        new_tags_for_sample: Set[TgtStringT] = set()
        for cur_tag in cur_phrases:
            tokenized_phrase = tokenizer.tokenize(cur_tag)
            n_tokens = len(tokenized_phrase)
            if n_tokens > max_words_tag:
                continue

            embedding = length_to_embedding_cache[n_tokens][cur_tag]

            near_tag_token_lengths: List[ItemsView[str, torch.Tensor]] = \
                [length_to_embedding_cache[tag_token_len].items() for
                 tag_token_len in length_to_embedding_cache.keys() if
                 abs(tag_token_len - n_tokens) <= similar_enough_count_diff]
            neighboring_length_tags = itertools.chain(*near_tag_token_lengths)
            for tag_to_apply, tag_to_apply_embedding in neighboring_length_tags:
                matched = False
                # Tag has already been added to the tags dict before loop
                if tag_to_apply == cur_tag:
                    matched = True
                elif similar_enough(
                        tag_to_apply=tokenizer.tokenize(tag_to_apply),
                        current_tag=tokenizer.tokenize(cur_tag)):
                    matched = True
                else:
                    cosine_threshold = cosine_threshold_phrase if len(
                        tokenized_phrase) > 1 else cosine_threshold_word
                    if cos(embedding,
                           tag_to_apply_embedding) > cosine_threshold:
                        matched = True
                if matched:
                    new_tags_for_sample.add(tag_to_apply)
        new_tags.append(list(new_tags_for_sample))
    return new_tags


if __name__ == '__main__':
    pass
