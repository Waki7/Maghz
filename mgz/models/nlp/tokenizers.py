# Load spacy tokenizer models, download them if they haven't been
# downloaded already
import os
from os.path import exists

import spacy
import torchtext.datasets as datasets
from spacy.language import Language
from spacy.tokens.doc import Doc
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe

from mgz.typing import *


def load_tokenizers() -> (Language, Language):
    try:
        spacy_de = spacy.load("de_core_news_sm")
    except IOError:
        os.system("python -m spacy download de_core_news_sm")
        spacy_de = spacy.load("de_core_news_sm")

    try:
        spacy_en = spacy.load("en_core_web_sm")
    except IOError:
        os.system("python -m spacy download en_core_web_sm")
        spacy_en = spacy.load("en_core_web_sm")

    return spacy_de, spacy_en


def tokenize(text: str, tokenizer: Language) -> List[str]:
    return [tok.text for tok in tokenizer.tokenizer(text)]


def yield_tokens(data_iter, tokenizer: Language, language_index: int):
    for from_to_tuple in data_iter:
        yield tokenize(from_to_tuple[language_index], tokenizer)


def build_vocabulary(spacy_de: Language, spacy_en: Language) -> (Vocab, Vocab):
    print("Building German Vocabulary ...")
    train: ShardingFilterIterDataPipe
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_src = build_vocab_from_iterator(
        yield_tokens(train + val + test, spacy_de, language_index=0),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    print("Building English Vocabulary ...")
    train, val, test = datasets.Multi30k(language_pair=("de", "en"))
    vocab_tgt = build_vocab_from_iterator(
        yield_tokens(train + val + test, spacy_en, language_index=1),
        min_freq=2,
        specials=["<s>", "</s>", "<blank>", "<unk>"],
    )

    vocab_src.set_default_index(vocab_src["<unk>"])
    vocab_tgt.set_default_index(vocab_tgt["<unk>"])

    return vocab_src, vocab_tgt


def load_vocab(spacy_de, spacy_en):
    if not exists("vocab.pt"):
        vocab_src, vocab_tgt = build_vocabulary(spacy_de, spacy_en)
        torch.save((vocab_src, vocab_tgt), "vocab.pt")
    else:
        vocab_src, vocab_tgt = torch.load("vocab.pt")
    print("Finished.\nVocabulary sizes:")
    print(len(vocab_src))
    print(len(vocab_tgt))
    return vocab_src, vocab_tgt


def main():
    # global variables used later in the script

    spacy_de, spacy_en = load_tokenizers()

    sentences: List[str] = ["hello! I am a sentence.", "I am another sentence."]
    out: List[Doc] = [spacy_en.tokenizer(sent) for sent in sentences]
    tokenized_sentences: List[List[str]] = [[tok.text for tok in doc] for doc in
                                            out]

    vocab_src, vocab_tgt = load_vocab(spacy_de, spacy_en)
    int_tokens: List[List[int]] = [vocab_src.forward(tokenized_sentence) for
                                   tokenized_sentence in
                                   tokenized_sentences]

if __name__ == '__main__':
    main()
