
import torchtext

def belu():
    torchtext.data.metrics.bleu_score(candidate_corpus: List[List[str]], references_corpus,
                                      max_n=4, weights=[0.25, 0.25, 0.25, 0.25])