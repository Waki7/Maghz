from __future__ import annotations

from functools import partial

from datasets import load_dataset
from datasets.arrow_dataset import Dataset
from datasets.dataset_dict import DatasetDict
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import spaces as sp
from mgz.ds.base_dataset import T, DataState, BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    SampleType, Sent2SentBatch
from mgz.typing import *

KEY_FOR_SAMPLE_TYPES = {
    SampleType.INPUT_TEXT: 'sources',
    SampleType.SUMMARY_TINY: 'summary/tiny',
    SampleType.SUMMARY_SHORT: 'summary/short',
    SampleType.SUMMARY_LONG: 'summary/long',
}


class MultiLexSum(SentenceDataset):
    '''
    Loading pattern for this class is that the parent class will specify the
    sample type for the parent class to load for it.
    '''

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen):
        super(MultiLexSum, self).__init__(tokenizer=tokenizer,
                                          max_src_len=max_src_len,
                                          max_tgt_len=max_tgt_len)
        ## main dataset information
        # A single element of this list will be:
        # {'sources': List[str], 'summary/long': str, 'summary/short': str, 'summary/tiny': str, 'id': str}

        self.dataset_keys: List[str] = []
        self.entry_keys: List[str] = []


    def __len__(self):
        'Denotes the total number of samples'
        return len(self.data)

    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        raise NotImplementedError

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        raise NotImplementedError(
            'There are implementations of this method in the subclasses')

    @property
    @overrides(SentenceDataset)
    def input_space(self) -> sp.Sentence:
        return sp.Sentence(
            len((self.vocab_src)), shape=(self.max_src_len,))

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        return sp.Sentence(len((self.vocab_tgt)),
                           shape=(self.max_tgt_len,))

    def __add__(self, other: Dataset[T]) -> 'ConcatDataset[T]':
        raise NotImplementedError

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2SentBatch.default_collate_fn, self, device)

    @overrides(BaseDataset)
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
            self.data_state = DataState.TRAIN
        elif val:
            examples: Dataset = multi_lexsum["validation"]
            self.data_state = DataState.VAL
        elif test:
            examples: Dataset = multi_lexsum["test"]
            self.data_state = DataState.TEST

        src_tgt_type: Tuple[SampleType, SampleType] = self._get_source_types()
        src_type, tgt_type = src_tgt_type
        sample: Dict[str, Union[SummaryT, SourceListT]]
        for sample in tqdm(examples):
            src_key_txt: str = KEY_FOR_SAMPLE_TYPES[src_type]
            tgt_key_txt: str = KEY_FOR_SAMPLE_TYPES[tgt_type]
            if sample[src_key_txt] is None or sample[tgt_key_txt] is None:
                continue
            self.data.append(
                {src_type: sample[src_key_txt], tgt_type: sample[tgt_key_txt]})
        logging.info('Loaded {} examples'.format(len(self.data)))

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
    for (entry) in tqdm(ds.data):
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
        entry = self.data[idx]
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
        entry = self.data[idx]
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
        entry = self.data[idx]
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
        ds.data[4]  # The first instance of the dev set
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
