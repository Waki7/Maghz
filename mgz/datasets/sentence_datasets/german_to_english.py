from functools import partial

import torchtext.datasets as datasets
from torch.utils.data import Dataset
from torch.utils.data.datapipes.iter.grouping import ShardingFilterIterDataPipe

import spaces as sp
from mgz.datasets.base_dataset import T
from mgz.datasets.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch
from mgz.models.nlp.tokenizers import load_tokenizers, load_vocab, tokenize
from mgz.typing import *


class GermanToEnglish(SentenceDataset):
    def __init__(self, max_length: int):
        super(GermanToEnglish, self).__init__()

        self.max_length = max_length
        self._data: List[Tuple[GermanT, EnglishT]] = []

        self.spacy_de, self.spacy_en = load_tokenizers()
        self.vocab_src, self.vocab_tgt = load_vocab(self.spacy_de,
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

    def __getitem__(self, idx):
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

    def _load(self, training: bool = False, validation: bool = False,
              testing: bool = False):
        iter: ShardingFilterIterDataPipe
        train_iter, valid_iter, test_iter = datasets.Multi30k(
            language_pair=("de", "en")
        )
        if training:
            iter = train_iter
        elif validation:
            iter = valid_iter
        elif testing:
            iter = test_iter
        self._data = list(iter)
        self.loaded = True

    def load_training_data(self):
        self._load(training=True)
        return self

    def load_validation_data(self):
        self._load(validation=True)
        return self

    def load_testing_data(self):
        self._load(testing=True)
        return self

    def cuda(self):
        self.use_cuda = True
        return self

    def cpu(self):
        self.use_cuda = False
        return self
