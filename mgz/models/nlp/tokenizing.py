# Load spacy tokenizer models, download them if they haven't been
# downloaded already
import os
from enum import Enum

import spacy
from spacy.language import Language
from spacy.tokens.doc import Doc

from mgz.typing import *


class TokenStrings(Enum):
    S = '[S]'
    EndS = '[/S]'
    BLANK = '[BLANK]'
    UNK = '[UNK]'
    SEP = '[SEP]'


class TokenIdxs(Enum):
    S = 0
    EndS = 1
    BLANK = 2
    UNK = 3
    SEP = 4


assert len(TokenStrings) == len(TokenIdxs)


class Tokenizer:
    @staticmethod
    def load_en_web_sm() -> Language:
        try:
            spacy_en = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            spacy_en = spacy.load("en_core_web_sm")
        return spacy_en

    @staticmethod
    def load_de_news_sm() -> Language:
        try:
            spacy_de = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            spacy_de = spacy.load("de_core_news_sm")
        return spacy_de

def main():
    from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum

    # global variables used later in the script
    spacy_de, spacy_en = Tokenizer.load_de_news_sm(), Tokenizer.load_en_web_sm()

    sentences: List[str] = [
        "hello! I am a sentence konbuna. <sep> [PAD] [SEP] Sentence",
        "I am another sentence."]
    out: List[Doc] = [spacy_en.tokenizer(sent) for sent in sentences]
    tokenized_sentences: List[List[str]] = [[tok.text for tok in doc] for doc in
                                            out]
    print('pre', tokenized_sentences[0])
    tokenized_sentences[0].append('[SEP]')
    print('post', tokenized_sentences[0])
    tokenized_sentences[0].append('[UNK]')
    tokenized_sentences[0].append('[22]')
    vocab_src, vocab_tgt = MultiLexSum.load_vocab(spacy_de, spacy_en)
    int_tokens: List[List[int]] = [vocab_src.forward(tokenized_sentence) for
                                   tokenized_sentence in
                                   tokenized_sentences]
    print(vocab_src.lookup_tokens([0, 1, 2, 3, 4, 5]))
    for wrd in int_tokens[0]:
        print(vocab_src.get_itos()[wrd])


if __name__ == '__main__':
    main()
