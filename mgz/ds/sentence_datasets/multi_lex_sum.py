from __future__ import annotations

from enum import Enum
from functools import partial

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from spacy.language import Language
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, PreTrainedTokenizerBase

import spaces as sp
from mgz.ds.base_dataset import T
from transformers import PreTrainedTokenizer

from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch, DataSplit, SampleType
from mgz.models.nlp.tokenizing import Tokenizer, TokenStrings
from mgz.typing import *


class InputSource(Enum):
    LONG = 'summary/long'
    SHORT = 'summary/short'
    TINY = 'summary/tiny'


class MultiLexSum(SentenceDataset):
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: SrcSeqLen):
        super(MultiLexSum, self).__init__()

        self.max_length = max_length
        self.split = None

        ## main dataset information
        self._data: List[
            Dict[str, Union[SummaryT, List[SourceListT]]]] = []

        self.dataset_keys: List[str] = []
        self.entry_keys: List[str] = []

        self.tokenizer_src: PreTrainedTokenizerBase = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizerBase = tokenizer
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_tgt.get_vocab()

        self.input_space = sp.Sentence(
            len((self.vocab_src)), shape=(max_length,))
        self.target_space = sp.Sentence(len((self.vocab_tgt)),
                                        shape=(max_length,))

        # --- Initialization flags ---
        self.use_cuda = False
        self.loaded = False

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._data)

    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        return self._data[idx]['sources'], self._data[idx][
            InputSource.LONG.value]

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
                flattened_source_text.append(self.tokenizer_src.sep_token)
            return flattened_source_text

        def tokenize_tgt(text: SummaryT) -> List[str]:
            return self.tokenizer_tgt.tokenize(text)

        def vocab_src(tokens: List[str]) -> List[int]:
            return [self.vocab_src[token] for token in tokens]

        def vocab_tgt(tokens: List[str]) -> List[int]:
            return [self.vocab_tgt[token] for token in tokens]

        return collate_batch(
            batch,
            tokenize_src,
            tokenize_tgt,
            vocab_src,
            vocab_tgt,
            device,
            max_padding=self.max_length,
            pad_id=self.tokenizer_src.pad_token_id
        )

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        iter: DatasetDict
        multi_lexsum = load_dataset("allenai/multi_lexsum", name="v20220616")
        self.dataset_keys = list(multi_lexsum.keys())
        self.entry_keys = multi_lexsum['train'][0].keys()
        # ['train', 'validation', 'test']
        # ['id', 'sources', 'summary/long', 'summary/short', 'summary/tiny']
        example: List[Dict[str, Union[SummaryT, SourceListT]]] = []
        if train:
            example: Dataset = multi_lexsum["train"]
            self.split = DataSplit.TRAIN
        elif val:
            example: Dataset = multi_lexsum["validation"]
            self.split = DataSplit.VAL
        elif test:
            example: Dataset = multi_lexsum["test"]
            self.split = DataSplit.TEST
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

    def pad_idx(self) -> int:
        return self.vocab_tgt[self.tokenizer_tgt.pad_token]

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)


def count_per_summary(ds: MultiLexSum):
    counts = {InputSource.LONG.value: 0, InputSource.SHORT.value: 0,
              InputSource.TINY.value: 0}
    for (entry) in tqdm(ds._data):
        if entry.get(InputSource.LONG.value) is not None:
            counts[InputSource.LONG.value] += 1
        if entry.get(InputSource.SHORT.value) is not None:
            counts[InputSource.SHORT.value] += 1
        if entry.get(InputSource.TINY.value) is not None:
            counts[InputSource.TINY.value] += 1
    print(counts)


def main():
    # please install HuggingFace ds by pip install ds
    ds = MultiLexSum(max_length=128).load_training_data()
    print(ds.tokenizer_src.tokenize("hello i am a test"))
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(tokenizer.tokenize("I have a new GPU!"))

    example: Dict = \
        ds._data[4]  # The first instance of the dev set
    print('dataset keys are: ', ds.dataset_keys)
    print('entry_keys keys are: ', ds.entry_keys)

    # keys are ['id', 'sources', 'summary/long', 'summary/short', 'summary/tiny']
    print('sources: \n', example['sources'], '\n')

    # for item in example['sources']:
    #     print(len(tokenize(item, Tokenizer.load_en_web_sm())))
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
    count_per_summary(ds)
    # for sum_len in ["long", "short", "tiny"]:
    #     print(example["summary/" + sum_len])  # Summaries of three lengths


if __name__ == '__main__':
    main()
