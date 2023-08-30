from __future__ import annotations

from collections import defaultdict

import torch.nn as nn
from tqdm import tqdm

import settings
from mgz.model_running.run_ops import embedding_controller
from mgz.model_vc.model_index import Indexer
from mgz.models.nlp.bart import BartModel
from mgz.models.nlp.metrics import word_count_diff
from mgz.typing import *


def tagging_with_semantic_grouping(training_phrases: List[List[SentenceT]],
                                   model_id: str = 'allenai/bart-large-multi_lexsum-short-tiny',
                                   cosine_threshold_phrase: float = 0.94,
                                   cosine_threshold_word: float = 0.88,
                                   similar_enough_count_diff: int = 1,
                                   max_words_tag: int = 5):
    def similar_enough(tag_to_apply: List[TokenT],
                       current_tag: List[TokenT]) -> bool:
        if len(current_tag) == 1 or abs(len(tag_to_apply) - len(
                current_tag)) > similar_enough_count_diff:
            return False
        if word_count_diff(tag_to_apply,
                           current_tag) <= similar_enough_count_diff:
            return True
        return False

    # Not ideal, but we need to do this to avoid GPU OOM
    orig_device = settings.DEVICE
    settings.DEVICE = torch.device('cpu')

    idxer = Indexer.get_default_index()
    model, tokenizer = idxer.get_cached_runtime_nlp_model(model_id, BartModel)
    model.eval()
    model.to(settings.DEVICE)
    cos = nn.CosineSimilarity(dim=-1, eps=1e-4)

    buckets: Dict[str, List[int]] = defaultdict(lambda: [])
    embedding_cache: Dict[str, torch.Tensor] = {}

    for sample_idx, phrases in tqdm(training_phrases):
        for cur_tag in phrases:
            embedding = embedding_cache.get(cur_tag,
                                            embedding_controller(model,
                                                                 cur_tag,
                                                                 tokenizer))
            tokenized_phrase = tokenizer.tokenize(cur_tag)

            if len(tokenized_phrase) < max_words_tag:
                buckets[cur_tag].append(sample_idx)
                embedding_cache[cur_tag] = embedding

            for tag_to_apply in buckets.keys():
                matched = False
                if tag_to_apply == cur_tag:
                    matched = True
                elif similar_enough(
                        tag_to_apply=tokenizer.tokenize(tag_to_apply),
                        current_tag=tokenizer.tokenize(cur_tag)):
                    matched = True
                elif len(tokenized_phrase) < max_words_tag:
                    matched = True
                else:
                    cosine_threshold = cosine_threshold_phrase if len(
                        tokenized_phrase) > 1 else cosine_threshold_word
                    if cos(embedding,
                           embedding_cache[tag_to_apply]) > cosine_threshold:
                        matched = True
                if matched:
                    buckets[tag_to_apply].append(sample_idx)
                    embedding_cache[tag_to_apply] = embedding

    settings.DEVICE = orig_device


if __name__ == '__main__':
    pass
