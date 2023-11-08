from __future__ import annotations

from enum import Enum
from functools import partial

from datasets.arrow_dataset import Dataset
from spacy.language import Language
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
import os
import spaces as sp
from mgz.ds.base_dataset import T
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch
from archive.models.tokenizing import TokenStrings
from mgz.typing import *


class InputSource(Enum):
    LONG = 'summary/long'
    SHORT = 'summary/short'
    TINY = 'summary/tiny'

DATASET_DIR = '../../../../datasets/20news-18828/'


class NewsGroup20(SentenceDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_length: SrcSeqLen):
        super(NewsGroup20, self).__init__()

        self.max_length = max_length
        self._data: List[Dict[str, Union[SummaryT, List[SourceListT]]]] = []

        self.tokenizer_src: PreTrainedTokenizer = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizer = tokenizer
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_src.get_vocab()

        self.input_space = sp.SentenceT(
            len((self.vocab_src)), shape=(max_length,))
        self.target_space = sp.SentenceT(len((self.vocab_tgt)),
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
    @overrides(SentenceDataset)
    def in_space(self) -> sp.SentenceT:
        return self.in_space

    @property
    @overrides(SentenceDataset)
    def pred_space(self) -> Union[sp.SentenceT, sp.RegressionTarget]:
        return self.target_space

    def __add__(self, other: Dataset[T]) -> 'ConcatDataset[T]':
        raise NotImplementedError

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    @staticmethod
    def yield_src_tokens(data_iter: Dataset,
                         tokenizer: Language) -> Generator[List[SourceListT]]:
        from_to_tuple: Dict[str, Union[str, List[str]]]
        for from_to_tuple in tqdm(data_iter):
            input_text: List[str] = from_to_tuple['sources']
            for text in input_text:
                yield tokenize(text, tokenizer)

    @staticmethod
    def yield_tgt_tokens(data_iter: Dataset,
                         tokenizer: Language) -> Generator[List[SummaryT]]:
        from_to_tuple: Dict[str, Union[str, List[str]]]
        for from_to_tuple in tqdm(data_iter):
            for key in ['summary/long', 'summary/short', 'summary/tiny']:
                tgt_text: str = from_to_tuple[key]
                if tgt_text is not None:
                    yield tokenize(tgt_text, tokenizer)

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[SourceListT, SummaryT]]):
        assert self.loaded, "Dataset not loaded"

        def tokenize_src(sources: SourceListT) -> List[str]:
            tokenized_sources: List[List[str]] = [
                self.tokenizer_src.tokenize(source_text) for
                source_text in
                sources]
            # We are going to flatten the list of lists and join them with a seperator token
            flattened_source_text: List[str] = []
            for tokenized_source in tokenized_sources:
                flattened_source_text.extend(tokenized_source)
                flattened_source_text.append(TokenStrings.SEP.value)
            return flattened_source_text

        def tokenize_tgt(text: SummaryT) -> List[str]:
            return self.tokenizer_tgt.tokenize(text)

        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            self.vocab_src,
            self.vocab_tgt,
            device,
            max_padding=self.max_length,
            pad_id=self.tokenizer_src.pad_token_id
        )

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        class_directories: List[str] = os.listdir(DATASET_DIR)
        class_directories.sort()
        print('DATASET_DIR', os.path.abspath(DATASET_DIR))
        print('class_directories', class_directories)
        n_tags = class_directories
        for n_tag, directory in enumerate(class_directories):
            class_dir = os.path.join(DATASET_DIR, directory)
            if directory.startswith('.') or not os.path.isdir(class_dir):
                print(class_dir)
                continue
            for file in os.listdir(class_dir):
                file_path = os.path.join(class_dir, file)
                if file.startswith('.') or not os.path.isfile(file_path):
                    continue
                with open(file_path) as f:
                    lines: List[str] = f.readlines()
                print(lines)
                exit(3)

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

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)


def main():
    # please install HuggingFace ds by pip install ds
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = NewsGroup20(tokenizer=tokenizer, max_length=128).load_training_data()
    print(ds.tokenizer_src.tokenize("hello i am a test"))
    print(tokenizer.tokenize("I have a new GPU!"))


if __name__ == '__main__':
    main()
