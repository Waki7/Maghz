from __future__ import annotations

from functools import partial
from os.path import exists

import torchtext.datasets as datasets
from spacy.language import Language
from torch.utils.data import Dataset
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab

import spaces as sp
from mgz.ds.base_dataset import T
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch
from mgz.models.nlp.tokenizing import Tokenizer, tokenize
from mgz.typing import *


class GermanToEnglish(SentenceDataset):
    def __init__(self, max_length: SrcSeqLen):
        super(GermanToEnglish, self).__init__()

        self.max_length = max_length
        self._data: List[Tuple[GermanT, EnglishT]] = []

        self.spacy_de, self.spacy_en = Tokenizer.load_de_news_sm(), Tokenizer.load_en_web_sm()
        self.vocab_src, self.vocab_tgt = self.load_vocab(self.spacy_de,
                                                         self.spacy_en)

        self.input_space = sp.Sentence(len(self.vocab_src), shape=(max_length,))
        self.target_space = sp.Sentence(len(self.vocab_tgt),
                                        shape=(max_length,))

        # --- Initialization flags ---
        self.use_cuda = False
        self.loaded = False

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._data)

    def __getitem__(self, idx) -> Tuple[GermanT, EnglishT]:
        return self._data[idx]

    @property
    def in_space(self) -> sp.Sentence:
        return self.in_space

    @property
    def pred_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        return self.target_space

    def __add__(self, other: Dataset[T]) -> 'ConcatDataset[T]':
        raise NotImplementedError

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[GermanT, EnglishT]]):
        assert self.loaded, "Dataset not loaded"

        def tokenize_de(text) -> List[str]:
            return tokenize(text, self.spacy_de)

        def tokenize_en(text) -> List[str]:
            return tokenize(text, self.spacy_en)

        return collate_batch(
            batch,
            tokenize_de,
            tokenize_en,
            self.vocab_src,
            self.vocab_tgt,
            device,
            max_padding=self.max_length,
            pad_id=self.vocab_src.get_stoi()["<blank>"],
        )

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def tokenize_de(self, text) -> List[str]:
        return tokenize(text, self.spacy_de)

    def tokenize_en(self, text) -> List[str]:
        return tokenize(text, self.spacy_en)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        iter: ShardingFilterIterDataPipe
        train_iter, valid_iter, test_iter = datasets.Multi30k(
            language_pair=("de", "en")
        )
        if train:
            iter = train_iter
        elif val:
            iter = valid_iter
        elif test:
            iter = test_iter
        self._data = list(iter)
        self.loaded = True

    def load_training_data(self):
        self._load(train=True)
        return self

    def load_validation_data(self):
        self._load(val=True)
        return self

    def load_testing_data(self):
        self._load(test=True)
        return self

    def cuda(self):
        self.use_cuda = True
        return self

    def cpu(self):
        self.use_cuda = False
        return self

    @staticmethod
    def yield_tokens(data_iter: List[Tuple[str, str]], tokenizer: Language,
                     language_index: int):
        from_to_tuple: Tuple[str, str]
        for from_to_tuple in data_iter:
            yield tokenize(from_to_tuple[language_index], tokenizer)

    @staticmethod
    def build_vocabulary(spacy_de: Language, spacy_en: Language) -> (
            Vocab, Vocab):
        print("Building German Vocabulary ...")
        train: ShardingFilterIterDataPipe
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))
        vocab_src = build_vocab_from_iterator(
            GermanToEnglish.yield_tokens(train + val + test, spacy_de,
                                         language_index=0),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        print("Building English Vocabulary ...")
        train, val, test = datasets.Multi30k(language_pair=("de", "en"))
        vocab_tgt = build_vocab_from_iterator(
            GermanToEnglish.yield_tokens(train + val + test, spacy_en,
                                         language_index=1),
            min_freq=2,
            specials=["<s>", "</s>", "<blank>", "<unk>"],
        )

        vocab_src.set_default_index(vocab_src["<unk>"])
        vocab_tgt.set_default_index(vocab_tgt["<unk>"])

        return vocab_src, vocab_tgt

    @staticmethod
    def load_vocab(spacy_de: Language, spacy_en: Language) -> (
            Vocab, Vocab):
        if not exists("vocab.pt"):
            vocab_src, vocab_tgt = GermanToEnglish.build_vocabulary(spacy_de,
                                                                    spacy_en)
            torch.save((vocab_src, vocab_tgt), "vocab.pt")
        else:
            vocab_src, vocab_tgt = torch.load("vocab.pt")
        print("Finished.\nVocabulary sizes:")
        print(len(vocab_src))
        print(len(vocab_tgt))
        return vocab_src, vocab_tgt
