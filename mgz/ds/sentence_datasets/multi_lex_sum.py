from __future__ import annotations

import logging
from functools import partial

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer
from transformers import PreTrainedTokenizerBase

import spaces as sp
from mgz.ds.base_dataset import T
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch, DataSplit, SampleType
from mgz.typing import *

KEY_FOR_SAMPLE_TYPES = {
    SampleType.INPUT_TEXT: 'sources',
    SampleType.SUMMARY_TINY: 'summary/tiny',
    SampleType.SUMMARY_SHORT: 'summary/short',
    SampleType.SUMMARY_LONG: 'summary/long',
}


class MultiLexSum(SentenceDataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen):
        super(MultiLexSum, self).__init__()

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        ## main dataset information
        # A single element of this list will be:
        # {'sources': List[str], 'summary/long': str, 'summary/short': str, 'summary/tiny': str, 'id': str}
        self._data: List[
            Dict[str, Union[
                SummaryT, List[SummaryT], SrcTextT, List[SrcTextT]]]] = []

        self.dataset_keys: List[str] = []
        self.entry_keys: List[str] = []

        self.tokenizer_src: PreTrainedTokenizerBase = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizerBase = tokenizer
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_tgt.get_vocab()

    def __len__(self):
        'Denotes the total number of samples'
        return len(self._data)

    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        raise NotImplementedError

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        raise NotImplementedError(
            'There are implementations of this method in the subclasses')

    @property
    def input_space(self) -> sp.Sentence:
        return sp.Sentence(
            len((self.vocab_src)), shape=(self.max_src_len,))

    @property
    def target_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        return sp.Sentence(len((self.vocab_tgt)),
                           shape=(self.max_tgt_len,))

    def __add__(self, other: Dataset[T]) -> 'ConcatDataset[T]':
        raise NotImplementedError

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[SrcTextT, TgtTextT]]):
        assert self.loaded, "Dataset not loaded"

        def tokenize_src(src_text: SrcTextT) -> List[str]:
            return self.tokenizer_tgt.tokenize(src_text)

        def tokenize_tgt(text: TgtTextT) -> List[str]:
            return self.tokenizer_tgt.tokenize(text)

        def vocab_src(tokens: List[str]) -> List[int]:
            return [self.vocab_src[token] for token in tokens]

        def vocab_tgt(tokens: List[str]) -> List[int]:
            return [self.vocab_tgt[token] for token in tokens]

        return collate_batch(
            batch=batch,
            src_tokenizer_pipeline=tokenize_src,
            tgt_tokenizer_pipeline=tokenize_tgt,
            src_vocab_pipeline=vocab_src,
            tgt_vocab_pipeline=vocab_tgt,
            device=device,
            pad_id=self.tokenizer_src.pad_token_id,
            max_src_len=self.max_src_len,
            max_tgt_len=self.max_tgt_len
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
        examples: List[Dict[str, Union[SummaryT, SourceListT]]] = []
        if train:
            examples: Dataset = multi_lexsum["train"]
            self.data_state = DataSplit.TRAIN
        elif val:
            examples: Dataset = multi_lexsum["validation"]
            self.data_state = DataSplit.VAL
        elif test:
            examples: Dataset = multi_lexsum["test"]
            self.data_state = DataSplit.TEST

        src_type: SampleType
        tgt_type: SampleType
        src_type, tgt_type = self._get_source_types()
        sample: Dict[str, Union[SummaryT, SourceListT]]
        for sample in tqdm(examples):
            if sample[KEY_FOR_SAMPLE_TYPES[tgt_type]] is None or sample[
                KEY_FOR_SAMPLE_TYPES[tgt_type]] is None:
                continue
            self._data.append(sample)
        logging.info('Loaded {} examples'.format(len(self._data)))

    def load_training_data(self):
        self._load(train=True)
        return self

    def gen_training_data(self):
        return self.__init__(self.tokenizer_src, self.max_src_len,
                             self.max_tgt_len).load_training_data()

    def load_validation_data(self):
        self._load(val=True)
        return self

    def gen_validation_data(self):
        return self.__init__(self.tokenizer_src, self.max_src_len,
                             self.max_tgt_len).load_validation_data()

    def load_testing_data(self):
        self._load(test=True)
        return self

    def gen_testing_data(self):
        return self.__init__(self.tokenizer_src, self.max_src_len,
                             self.max_tgt_len).load_testing_data()

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
    counts = {SampleType.SUMMARY_LONG: 0,
              SampleType.SUMMARY_SHORT: 0,
              SampleType.SUMMARY_TINY: 0}
    for (entry) in tqdm(ds._data):
        if entry.get(SampleType.SUMMARY_LONG) is not None:
            counts[SampleType.SUMMARY_LONG] += 1
        if entry.get(SampleType.SUMMARY_SHORT) is not None:
            counts[SampleType.SUMMARY_SHORT] += 1
        if entry.get(SampleType.SUMMARY_TINY) is not None:
            counts[SampleType.SUMMARY_TINY] += 1
    print(counts)


# table for these stats: https://github.com/multilexsum/dataset

class MultiLexSumLongToTiny(MultiLexSum):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_position_embeddings: int):
        assert max_position_embeddings >= 1024
        super(MultiLexSumLongToTiny, self).__init__(tokenizer, 1024,
                                                    256)

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        return SampleType.SUMMARY_LONG, SampleType.SUMMARY_TINY

    @overrides(MultiLexSum)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        entry = self._data[idx]
        return entry[SampleType.SUMMARY_LONG.value], entry[
            SampleType.SUMMARY_TINY.value]


class MultiLexSumLongToShort(MultiLexSum):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_position_embeddings: int):
        assert max_position_embeddings >= 1024
        super(MultiLexSumLongToShort, self).__init__(tokenizer, 1024,
                                                     256)

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        return SampleType.SUMMARY_LONG, SampleType.SUMMARY_SHORT

    @overrides(MultiLexSum)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        entry = self._data[idx]
        return entry[SampleType.SUMMARY_LONG], entry[
            SampleType.SUMMARY_SHORT]


class MultiLexSumShortToTiny(MultiLexSum):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_position_embeddings: int):
        assert max_position_embeddings >= 1024
        super(MultiLexSumShortToTiny, self).__init__(tokenizer, 1024,
                                                     128)

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        return SampleType.SUMMARY_SHORT, SampleType.SUMMARY_TINY

    @overrides(MultiLexSum)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        entry = self._data[idx]
        return entry[SampleType.SUMMARY_LONG], entry[
            SampleType.SUMMARY_SHORT]


def main():
    # please install HuggingFace ds by pip install ds
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = MultiLexSumShortToTiny(tokenizer=tokenizer,
                                max_position_embeddings=1024).load_training_data()

    print('----Tokenization Example: ',
          ds.tokenizer_src.tokenize("hello i am a test"))

    example: Dict = \
        ds._data[4]  # The first instance of the dev set
    print('dataset keys are: ', ds.dataset_keys)
    print('entry_keys keys are: ', ds.entry_keys)
    print(type(example['sources']))
    print(type(example['summary/short']))

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
