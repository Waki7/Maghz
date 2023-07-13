import torchtext
from mgz.typing import *
from torchtext.data.metrics import bleu_score

def bleu_from_batch_decode(candidate_corpus: List[str], references_corpus: List[str]):
    return bleu_score(candidate_corpus, references_corpus)

def bleu(candidate_corpus: List[List[str]],
         references_corpus: List[List[List[str]]], max_n=4,
         weights=[0.25, 0.25, 0.25, 0.25]):
    return bleu_score(candidate_corpus, references_corpus, max_n=max_n,
                      weights=weights)


def main():
    candidate_corpus = [['My', 'full', 'pytorch', 'test'],
                        ['Another', 'Sentence']]
    # candidate_corpus = [['My', 'full', 'pytorch', 'test'],
    #                     ['No', 'Match']]
    references_corpus = [
        [['My', 'full', 'pytorch', 'test'],
         ['Completely', 'Different']],
        [['No', 'Match']]
    ]

    # references_corpus =  [['My', 'full', 'pytorch', 'test'],
    #                     ['Another', 'Sentence']]
    score = bleu(candidate_corpus, references_corpus)
    print(score)


if __name__ == '__main__':
    main()
