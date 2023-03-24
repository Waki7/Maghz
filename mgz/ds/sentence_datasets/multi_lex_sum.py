from __future__ import annotations

from functools import partial
from os.path import exists

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from spacy.language import Language
from torch.utils.data import Dataset
from torchtext.vocab import build_vocab_from_iterator
from torchtext.vocab.vocab import Vocab
from tqdm import tqdm

import spaces as sp
from mgz.ds.base_dataset import T
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch
from mgz.models.nlp.tokenizing import Tokenizer, tokenize, TokenStrings
from mgz.typing import *


class MultiLexSum(SentenceDataset):
    def __init__(self, max_length: SrcSeqLen):
        super(MultiLexSum, self).__init__()

        self.max_length = max_length
        self._data: List[Dict[str, Union[SummaryT, List[SourceListT]]]] = []

        self.tokenizer_src = Tokenizer.load_en_web_sm()
        self.tokenizer_tgt = Tokenizer.load_en_web_sm()
        self.vocab_src, self.vocab_tgt = self.load_vocab(self.tokenizer_src,
                                                         self.tokenizer_tgt)

        self.input_space = sp.Sentence(len(self.vocab_src), shape=(max_length,))
        self.target_space = sp.Sentence(len(self.vocab_tgt),
                                        shape=(max_length,))

        # --- Initialization flags ---
        self.use_cuda = False
        self.loaded = False

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._data)

    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        return self._data[idx]['sources'], self._data[idx]['summary/long']

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

        def tokenize_src(sources: SourceListT) -> List[str]:
            tokenized_sources: List[List[str]] = [
                tokenize(source_text, self.tokenizer_src) for source_text in
                sources]
            # We are going to flatten the list of lists and join them with a seperator token
            flattened_source_text: List[str] = []
            for tokenized_source in tokenized_sources:
                flattened_source_text.extend(tokenized_source)
                flattened_source_text.append(TokenStrings.SEP.value)
            return flattened_source_text

        def tokenize_tgt(text: SummaryT) -> List[str]:
            return tokenize(text, self.tokenizer_tgt)

        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            self.vocab_src,
            self.vocab_tgt,
            device,
            max_padding=self.max_length,
            pad_id=self.vocab_src.get_stoi()[TokenStrings.BLANK.value],
        )

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def tokenize_src(self, text) -> List[str]:
        return tokenize(text, self.tokenizer_src)

    def tokenize_tgt(self, text) -> List[str]:
        return tokenize(text, self.tokenizer_tgt)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        iter: DatasetDict
        multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")
        # ['train', 'validation', 'test']
        # ['id', 'sources', 'summary/long', 'summary/short', 'summary/tiny']
        example: List[Dict[str, Union[SummaryT, SourceListT]]] = []
        if train:
            example: Dataset = multi_lexsum["train"]
        elif val:
            example: Dataset = multi_lexsum["validation"]
        elif test:
            example: Dataset = multi_lexsum["test"]
        self._data = example
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
    def yield_src_tokens(data_iter: Dataset,
                         tokenizer: Language):
        from_to_tuple: Dict[str, Union[str, List[str]]]
        for from_to_tuple in tqdm(data_iter):
            input_text: List[str] = from_to_tuple['sources']
            for text in input_text:
                yield tokenize(text, tokenizer)

    @staticmethod
    def yield_tgt_tokens(data_iter: Dataset,
                         tokenizer: Language):
        from_to_tuple: Dict[str, Union[str, List[str]]]
        for from_to_tuple in tqdm(data_iter):
            for key in ['summary/long', 'summary/short', 'summary/tiny']:
                tgt_text: str = from_to_tuple[key]
                if tgt_text is not None:
                    yield tokenize(tgt_text, tokenizer)

    @staticmethod
    def build_vocabulary(tokenizer_src: Language,
                         tokenizer_tgt: Language) -> (Vocab, Vocab):
        print("Building English Vocabulary ...")
        multi_lexsum: DatasetDict = load_dataset("allenai/multi_lexsum",
                                                 name="v20220616")
        src: Dataset = multi_lexsum["train"]
        vocab_src = build_vocab_from_iterator(
            MultiLexSum.yield_src_tokens(src, tokenizer_src),
            min_freq=2,
            specials=["[S]", "[/S]", "[BLANK]", "[UNK]", "[SEP]"],
        )

        vocab_tgt = build_vocab_from_iterator(
            MultiLexSum.yield_tgt_tokens(src, tokenizer_tgt),
            min_freq=2,
            specials=["[S]", "[/S]", "[BLANK]", "[UNK]", "[SEP]"],
        )

        vocab_src.set_default_index(vocab_src[TokenStrings.UNK.value])
        vocab_tgt.set_default_index(vocab_tgt[TokenStrings.UNK.value])

        return vocab_src, vocab_tgt

    @staticmethod
    def load_vocab(spacy_de: Language, spacy_en: Language) -> (
            Vocab, Vocab):
        vocab_path = "C:/Users/ceyer/OneDrive/Documents/Projects/Maghz/" \
                     "index_dir/vocab_storage/multi_lex_sum_vocab.pt"
        if not exists(vocab_path):
            vocab_src, vocab_tgt = MultiLexSum.build_vocabulary(spacy_de,
                                                                spacy_en)
            torch.save((vocab_src, vocab_tgt), vocab_path)
        else:
            vocab_src, vocab_tgt = torch.load(vocab_path)
        print("Finished.\nVocabulary sizes:")
        print(len(vocab_src))
        print(len(vocab_tgt))
        return vocab_src, vocab_tgt


def main():
    # please install HuggingFace ds by pip install ds

    multi_lexsum: DatasetDict = load_dataset("allenai/multi_lexsum",
                                             name="v20220616")
    print(type(multi_lexsum))
    # Download multi_lexsum locally and load it as a Dataset object

    example: Dict = \
        multi_lexsum["validation"][4]  # The first instance of the dev set
    print(type(multi_lexsum["validation"]))
    print(type(multi_lexsum["validation"][0]))
    print(list(multi_lexsum.keys()))
    print(list(example.keys()))
    for item in example['sources']:
        print(len(tokenize(item, Tokenizer.load_en_web_sm())))
    for a, b in example.items():
        if b is not None:
            if isinstance(b, str):
                print(a, ": ", type(b), " length of ", len(b), " content: ",
                      b[:100],
                      "... ")
            else:
                print(a, ": ", type(b), " length of  ", len(b))
        else:
            print(a, ": ", type(b), ".... ", b)

    # for sum_len in ["long", "short", "tiny"]:
    #     print(example["summary/" + sum_len])  # Summaries of three lengths


if __name__ == '__main__':
    main()
