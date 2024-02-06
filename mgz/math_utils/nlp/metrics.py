from __future__ import annotations

import torch.nn as nn
from torchtext.data.metrics import bleu_score

import mgz.settings as settings
from mgz.models.nlp.bart import BartModel
from mgz.typing import *
from mgz.version_control.model_index import Indexer


class DistanceMeasuresPerClass:
    """
    A class for calculating distance measures between sentences.
    """

    @staticmethod
    def euclidean_distance(class_embeddings: FloatTensorT['NClasses,EmbedLen'],
                           query_embeddings: FloatTensorT['NQuery,EmbedLen']) -> \
            FloatTensorT['NQuery,NClasses']:
        distance_to_classes: FloatTensorT['NQuery,NClasses'] = \
            torch.linalg.norm(
                query_embeddings[:, None, :] - class_embeddings[None, :, :],
                dim=-1, ord=2)
        return distance_to_classes

    @staticmethod
    def cosine_similarity(class_embeddings: FloatTensorT['NClasses,EmbedLen'],
                          query_embeddings: FloatTensorT['NQuery,EmbedLen']) -> \
            FloatTensorT['NQuery,NClasses']:
        query_n, class_n = query_embeddings.norm(dim=-1)[:,
                           None], class_embeddings.norm(dim=-1)[:, None]
        query_norm = query_embeddings / torch.max(query_n,
                                                  1.0e-4 * torch.ones_like(
                                                      query_n))
        class_norm = class_embeddings / torch.max(class_n,
                                                  1.0e-4 * torch.ones_like(
                                                      class_n))
        distance_to_classes: FloatTensorT['NQuery,NClasses'] = \
            torch.mm(query_norm, class_norm.transpose(0, 1))
        return distance_to_classes

    # @staticmethod
    # def cosine_similarity(class_embeddings: FloatTensorT['NClasses,EmbedLen'],
    #                       query_embeddings: FloatTensorT['NQuery,EmbedLen']) -> \
    #         FloatTensorT['NQuery,NClasses']:
    #     return pairwise_cosine_similarity(query_embeddings, class_embeddings)

    @staticmethod
    def inner_dot_product(class_embeddings: FloatTensorT['NClasses,EmbedLen'],
                          query_embeddings: FloatTensorT['NQuery,EmbedLen']) -> \
            FloatTensorT['NQuery,NClasses']:
        return torch.mm(query_embeddings, class_embeddings)


def bleu_from_tokenized_sentences(candidate_corpus: List[TokenT],
                                  references_corpus: List[TokenT],
                                  max_n: int = 4,
                                  weights: Optional[
                                      List[float]] = None) -> float:
    """
    Calculate BLEU score between a candidate sentence and a reference sentence.

    Args:
        candidate_corpus (List[TokenT]): Tokenized candidate sentence.
        references_corpus (List[TokenT]): Tokenized reference sentence(s).
        max_n (int, optional): Maximum n-gram order for BLEU calculation. Defaults to 4.
        weights (List[float], optional): Weights for individual n-grams in BLEU calculation.
            Defaults to [1 / max_n] * max_n.

    Returns:
        float: BLEU score.
    """
    if weights is None:
        weights = [1 / max_n] * max_n
    print('candidate_corpus', candidate_corpus)
    print('references_corpus', references_corpus)
    return bleu_score([candidate_corpus], [[references_corpus]], max_n=max_n,
                      weights=weights)


def cosine_similarity_from_raw_sentences(sentences1: List[SentenceT],
                                         sentences2: List[SentenceT],
                                         model_id: str = 'allenai/bart-large-multi_lexsum-short-tiny') -> \
        List[float]:
    """
    Calculate cosine similarity between embeddings of two lists of sentences using a pre-trained model.

    Args:
        sentences1 (List[SentenceT]): List of sentences for comparison.
        sentences2 (List[SentenceT]): List of sentences for comparison.
        model_id (str, optional): Identifier for the pre-trained model. Defaults to 'allenai/bart-large-multi_lexsum-short-tiny'.

    Returns:
        List[float]: List of cosine similarity scores for each pair of sentences.
    """
    from mgz.model_running.run_ops import embedding_controller_from_texts

    idxer = Indexer.get_default_index()
    model, tokenizer = idxer.get_cached_runtime_nlp_model(model_id, BartModel)
    model.eval()
    model.to(settings.DEVICE)
    embedding1 = embedding_controller_from_texts(model, sentences1, tokenizer)
    embedding2 = embedding_controller_from_texts(model, sentences2, tokenizer)
    cos = nn.CosineSimilarity(dim=-1, eps=1e-4)
    return cos(embedding1, embedding2).cpu().tolist()


def bleu(candidate_corpus: List[List[str]],
         references_corpus: List[List[List[str]]], max_n: int = 4,
         weights: Optional[List[float]] = None) -> float:
    """
    Calculate BLEU score between candidate and reference sentence(s).

    Args:
        candidate_corpus (List[List[str]]): List of candidate sentences.
        references_corpus (List[List[List[str]]]): List of reference sentences for comparison.
        max_n (int, optional): Maximum n-gram order for BLEU calculation. Defaults to 4.
        weights (List[float], optional): Weights for individual n-grams in BLEU calculation.
            Defaults to [1 / max_n] * max_n.

    Returns:
        float: BLEU score.
    """
    if weights is None:
        weights = [1 / max_n] * max_n
    return bleu_score(candidate_corpus, references_corpus, max_n=max_n,
                      weights=weights)


def word_count_diff(src_words: List[TokenT], tgt_words: List[TokenT]) -> int:
    """
    Calculate the difference in word count between two lists of tokenized words.

    Args:
        src_words (List[TokenT]): List of tokenized words from the source.
        tgt_words (List[TokenT]): List of tokenized words from the target.

    Returns:
        int: Absolute difference in word count between the source and target lists.
    """
    diff_words = set(src_words).difference(set(tgt_words))
    diff_words_rev = set(tgt_words).difference(set(src_words))
    return len(diff_words) + len(diff_words_rev)


####################################################################
########################## INVESTIGATIONS #########################
####################################################################
def investigate_cosine():
    tokenized1 = ['calderbank letter']
    tokenized2 = ['calderbank letter', 'another sentence']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['great']
    tokenized2 = ['good']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['bye']
    tokenized2 = ['good']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['not']
    tokenized2 = ['good']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['big']
    tokenized2 = ['large']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['meat']
    tokenized2 = ['chicken']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))

    tokenized1 = ['meat']
    tokenized2 = ['vegetables']
    print(cosine_similarity_from_raw_sentences(tokenized1, tokenized2))


def investigate_bleu():
    candidate_corpus = [['Completely', 'Different'],  # <- call this A
                        ['No', 'Match']]  # <- call this B
    references_corpus = [
        [['My', 'full', 'pytorch', 'test'],  # <- A needs to match this
         ['Different', 'but', 'four', 'words']],  # or A needs to match this
        [['No', 'Match' 'should', 'match', 'this']]  # <- B needs to match this
    ]

    candidate_corpus = [['Completely', 'Different'],  # <- call this A
                        ['No', 'Match']]  # <- call this B
    references_corpus = [
        [['My', 'full', 'pytorch', 'test'],  # <- A needs to match this
         ],
        [['No']]  # <- B needs to match this
    ]

    for (candidate, refs) in zip(candidate_corpus, references_corpus):
        print(candidate)
        print(refs)
    # references_corpus =  [['My', 'full', 'pytorch', 'test'],
    #                     ['Another', 'Sentence']]
    score = bleu(candidate_corpus, references_corpus, max_n=2, )
    print(score)


####################################################################

def main():
    investigate_cosine()


if __name__ == '__main__':
    main()
