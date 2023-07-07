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

import os
import spaces as sp
from mgz.ds.base_dataset import T
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch, SampleType
from mgz.models.nlp.tokenizing import Tokenizer, TokenStrings
from mgz.typing import *
from bs4 import BeautifulSoup, ResultSet

PATH = '../../../../datasets/corpus'


class AusCaseReports(SentenceDataset):
    def __init__(self, max_length: SrcSeqLen):
        super(AusCaseReports, self).__init__()

        self.max_length = max_length
        self._data: List[Dict[str, Union[SummaryT, List[SourceListT]]]] = []

        self.tokenizer_src: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "facebook/bart-base")
        self.tokenizer_tgt: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
            "facebook/bart-base")
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_src.get_vocab()

        self.input_space = sp.Sentence(
            len((self.vocab_src)), shape=(max_length,))
        self.target_space = sp.Sentence(len((self.vocab_tgt)),
                                        shape=(max_length,))

        # --- Initialization flags ---
        self.use_cuda = False
        self.loaded = False
        self.training_ratio = .7
        self.validation_ratio = .15
        self.test_ratio = .15

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

        # Read all the input text files, from readme:
        # 1. fulltext: Contains the full text and the catchphrases of all the cases from the FCA. Every document (<case>) contains:
        # 	<name> : name of the case
        # 	<AustLII> : link to the austlii page from where the document was taken
        # 	<catchphrases> : contains a list of <catchphrase> elements
        # 		<catchphrase> : a catchphrase for the case, with an id attribute
        # 	<sentences> : contains a list of <sentence> elements
        # 		<sentence> : a sentence with id attribute
        fulltext_dir = os.path.join(PATH, 'fulltext')

        # Read all the input text files, from readme:
        # 2. citations_summ: Contains citations element for each case. Fields:
        # 	<name> : name of the case
        # 	<AustLII> : link to the austlii page from where the document was taken
        # 	<citphrases> : contains a list of <citphrase> elements
        # 		<citphrase> : a citphrase for the case, this is a catchphrase from a case which is cited or cite the current one. Attributes: id,type (cited or citing),from(the case from where the catchphrase is taken).
        # 	<citances> : contains a list of <citance> elements
        # 		<citance> : a citance for the case, this is a sentence from a later case that mention the current case. Attributes: id,from(the case from where the citance is taken).
        # 	<legistitles> : contains a list of <title> elements
        # 		<title> : Title of a piece of legislation cited by the current case (can be an act or a specific section).
        citation_sum_dir = os.path.join(PATH, 'citations_summ')

        # 3. citations_class: Contains for each case a list of labeled citations. Fields:
        # 	<name> : name of the case
        # 	<AustLII> : link to the austlii page from where the document was taken
        # 	<citations> : contains a list of <citation> elements
        # 		<citation> : a citation to an older case, it has an id attribute and contains the following elements:
        # 			<class> : the class of the citation as indicated on the document
        # 			<tocase> : the name of the case which is cited
        # 			<AustLII> : the link to the document of the case which is cited
        # 			<text> : paragraphs in the cited case where the current case is mentioned
        citation_cls_dir = os.path.join(PATH, 'citations_class')

        total_samples = len(os.listdir(fulltext_dir))
        n_train = (int)(total_samples * self.training_ratio)
        n_val = (int)(total_samples * self.validation_ratio)
        n_test = (int)(total_samples * self.test_ratio)
        start_end: Tuple[int, int] = None
        if train:
            start_end = range(0, n_train)
        elif val:
            start_end = range(n_train, n_train + n_val)
        elif test:
            start_end = range(total_samples - n_test, total_samples)
        if start_end is None:
            raise ValueError("No dataset split selected")

        keys = []
        data_entries = {SampleType.NAME: [], SampleType.CATCHPHRASES: [],
                        SampleType.INPUT_TEXT: []}
        for file in os.listdir(fulltext_dir)[start_end[0]: start_end[1]]:
            file = os.path.join(fulltext_dir, file)
            with open(file) as fp:
                soup = BeautifulSoup(fp, 'html.parser')
                case_name_xml: ResultSet = soup.find('name')
                catchphrases_xml: List[ResultSet] = soup.find_all('catchphrase')
                sentences_xml: List[ResultSet] = soup.find_all('sentence')

                data_entries[SampleType.NAME].append(case_name_xml.text)
                data_entries[SampleType.CATCHPHRASES].append(
                    [entry.text for entry in catchphrases_xml])
                data_entries[SampleType.INPUT_TEXT].append(
                    [entry.text for entry in sentences_xml])

                if file.endswith(".txt"):
                    keys.append(file.split(".")[0])

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

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)

    def gen(self) -> Generator[T, None, None]:
        raise NotImplementedError


def main():
    # please install HuggingFace ds by pip install ds
    ds = AusCaseReports(max_length=256).load_testing_data()

    exit(3)
    print(ds.tokenizer_src.tokenize("hello i am a test"))
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    print(tokenizer.tokenize("I have a new GPU!"))
    multi_lexsum: DatasetDict = load_dataset("allenai/multi_lexsum",
                                             name="v20220616")

    example: Dict = \
        multi_lexsum["validation"][4]  # The first instance of the dev set
    print(list(multi_lexsum.keys()))
    # keys are ['id', 'sources', 'summary/long', 'summary/short', 'summary/tiny']
    print(list(example.keys()))
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

    # for sum_len in ["long", "short", "tiny"]:
    #     print(example["summary/" + sum_len])  # Summaries of three lengths


if __name__ == '__main__':
    main()