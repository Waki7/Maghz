import torch.nn as nn
from torchtext.data.metrics import bleu_score

import settings
from mgz.model_vc.model_index import Indexer
from mgz.models.nlp.bart import BartModel
from mgz.typing import *


def bleu_from_tokenized_sentences(candidate_corpus: List[TokenT],
                                  references_corpus: List[TokenT],
                                  max_n: int = 4,
                                  weights: List[float] = None):
    if weights is None:
        weights = [1 / max_n] * max_n
    return bleu_score([candidate_corpus], [[references_corpus]], max_n=max_n,
                      weights=weights)


def cosine_similarity_from_raw_sentences(sentences1: List[SentenceT],
                                         sentences2: List[SentenceT],
                                         model_id: str = 'allenai/bart-large-multi_lexsum-short-tiny') -> \
        List[float]:
    from mgz.model_running.run_ops import embedding_controller
    # TODO this is not how we should be doing it in the future
    idxer = Indexer.get_default_index()
    model, tokenizer = idxer.get_cached_runtime_nlp_model(model_id, BartModel)
    model.eval()
    model.to(settings.DEVICE)
    embedding1 = embedding_controller(model, sentences1, tokenizer)
    embedding2 = embedding_controller(model, sentences2, tokenizer)
    cos = nn.CosineSimilarity(dim=-1, eps=1e-4)
    return cos(embedding1, embedding2).cpu().tolist()


def bleu(candidate_corpus: List[List[str]],
         references_corpus: List[List[List[str]]], max_n=4, weights=None):
    '''
    Each Sentence in candidate_corpus will be compared to each List[Sentence] in
    references_corpus. Needs to have at least max_n words for it to be counted.
    '''
    if weights is None:
        weights = [1 / max_n] * max_n
    return bleu_score(candidate_corpus, references_corpus, max_n=max_n,
                      weights=weights)


def word_count_diff(src_words: List[TokenT], tgt_words: List[TokenT]):
    diff_words = set(src_words).difference(set(tgt_words))
    diff_words_rev = set(tgt_words).difference(set(src_words))
    return len(diff_words) + len(diff_words_rev)


####################################################################
def investigate_cosine():
    tokenized1 = ['calderbank letter']
    tokenized2 = ['calderbank letter']
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
