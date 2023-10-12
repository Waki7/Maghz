from __future__ import annotations

import os
import re
from functools import partial
from pathlib import Path

import torch.utils.data
from bs4 import BeautifulSoup, ResultSet
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    Sent2SentBatch, SampleType
from mgz.typing import *

DATASET_DIR = os.path.join(
    Path(__file__).resolve().parent.parent.parent.parent,
    'datasets/enron_with_categories/').replace("\\", "/")
CATEGORIES = {
    1: {
        1: "Company Business, Strategy, etc.",
        2: "Purely Personal",
        3: "Personal but in professional context",
        4: "Logistic Arrangements",
        5: "Employment arrangements",
        6: "Document editing/checking (collaboration)",
        7: "Empty message (due to missing attachment)",
        8: "Empty message",
    },
    2: {
        1: "Includes new text in addition to forwarded material",
        2: "Forwarded emails including replies",
        3: "Business letters / documents",
        4: "News articles",
        5: "Government / academic reports",
        6: "Government actions (such as results of a hearing, etc)",
        7: "Press releases",
        8: "Legal documents (complaints, lawsuits, advice)",
        9: "Pointers to urls",
        10: "Newsletters",
        11: "Jokes, humor (related to business)",
        12: "Jokes, humor (unrelated to business)",
        13: "Attachment(s) (assumed missing)",
    },
    3: {
        1: "regulations and regulators (includes price caps)",
        2: "internal projects -- progress and strategy",
        3: "company image -- current",
        4: "company image -- changing / influencing",
        5: "political influence / contributions / contacts",
        6: "california energy crisis / california politics",
        7: "internal company policy",
        8: "internal company operations",
        9: "alliances / partnerships",
        10: "legal advice",
        11: "talking points",
        12: "meeting minutes",
        13: "trip reports",
    },
    4: {
        1: "jubilation",
        2: "hope / anticipation",
        3: "humor",
        4: "camaraderie",
        5: "admiration",
        6: "gratitude",
        7: "friendship / affection",
        8: "sympathy / support",
        9: "sarcasm",
        10: "secrecy / confidentiality",
        11: "worry / anxiety",
        12: "concern",
        13: "competitiveness / aggressiveness",
        14: "triumph / gloating",
        15: "pride",
        16: "anger / agitation",
        17: "sadness / despair",
        18: "shame",
        19: "dislike / scorn",
    },
}


class LoadingTask:
    '''
    This class is used to load additional categories that were generated programmatically as additional tasks or pretext tasks.
    '''

    def __init__(self):
        pass


class EnronEmails(SentenceDataset):
    '''
    Loading pattern for this class is that the parent will load all the data
    and the children will load a subset of a modified subset of the data.
    There are children that are also meta learning datasets.
    '''

    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen,
                 training_ratio=0.7):  # change for testing/verification
        super(EnronEmails, self).__init__(tokenizer=tokenizer,
                                          max_src_len=max_src_len,
                                          max_tgt_len=max_tgt_len)
        # --- Initialization flags ---
        self.training_ratio = training_ratio
        self.validation_ratio = .15
        self.test_ratio = .15

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
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
    def target_space(self) -> sp.Sentence:
        return sp.Sentence(len((self.vocab_tgt)),
                           shape=(self.max_tgt_len,))

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2SentBatch.default_collate_fn, self, device)

    def _pre_load_set_state_data_range(self, train: bool,
                                       val: bool,
                                       test: bool) -> Tuple[
        List[FilePath], List[FilePath]]:
        n_files = len(os.listdir(os.path.join(DATASET_DIR, 'fulltext')))
        n_train = (int)(n_files * self.training_ratio)
        n_val = (int)(n_files * self.validation_ratio)
        n_test = (int)(n_files * self.test_ratio)
        sample_range = range(0, n_files)
        if train:
            sample_range = range(0, n_train)
            self.data_state = DataState.TRAIN
        elif val:
            sample_range = range(n_train, n_train + n_val)
            self.data_state = DataState.VAL
        elif test:
            sample_range = range(n_files - n_test, n_files)
            self.data_state = DataState.TEST
        if sample_range is None:
            raise ValueError("No dataset split selected")
        return sample_range

    @overrides(BaseDataset)
    def _load(self, train: bool = False, val: bool = False, test: bool = False):

        email_file_paths, label_file_paths = self._pre_load_paths_data_range(
            train, val, test)
        logging.info('Loading data from: ' + DATASET_DIR)
        email_file_paths: List[FilePath] = []
        label_file_paths: List[FilePath] = []
        directories = [os.path.join(DATASET_DIR, str(i)) for i in range(1, 9)]
        for dir in directories:
            for file in os.listdir(dir):
                if file.endswith('.txt'):
                    email_file_paths.append(os.path.join(dir, file))
                if file.endswith('.cats'):
                    label_file_paths.append(os.path.join(dir, file))
        assert len(email_file_paths) == len(
            label_file_paths), 'expect lengths to be the same instead they are {} vs {}'.format(
            len(email_file_paths), len(label_file_paths))
        for email_path, label_path in tqdm(
                zip(email_file_paths, label_file_paths)):
            print(label_path)
            with open(label_path) as f:
                # label shows up as 'int,int,int\n'
                labels: List[str] = f.readlines()
                categories: List[Tuple[int, ...]] = [
                    Tuple(map(int, re.findall(r'\d+', label))) for
                    label in labels]
                tags_for_email: List[str] = \
                    [CATEGORIES[c[0]][c[1]] for c in categories]

            print(categories)

            def get_text_from_email(msg):
                '''To get the content from email objects'''
                parts = []
                for part in msg.walk():
                    if part.get_content_type() == 'text/plain':
                        parts.append(part.get_payload())
                return ''.join(parts)

            def split_email_addresses(line):
                '''To separate multiple email addresses'''
                if line:
                    addrs = line.split(',')
                    addrs = frozenset(map(lambda x: x.strip(), addrs))
                else:
                    addrs = None
                return addrs

            data_entry: Dict[SampleType, Union[str, List[str]]] = \
                {SampleType.KEY: None,
                 SampleType.NAME: None,
                 SampleType.CATCHPHRASES: [],
                 SampleType.INPUT_TEXT: []}
            try:
                soup = BeautifulSoup(fp, 'html.parser')
            except UnicodeDecodeError as e:
                logging.info(
                    "Error parsing file non utf-8 character found: " + file)
                continue
            case_name_xml: ResultSet = soup.find('name')
            catchphrases_xml: List[ResultSet] = soup.find_all(
                'catchphrase')
            sentences_xml: List[ResultSet] = soup.find_all('sentence')

            data_entry[SampleType.KEY] = (file.split(".")[0])
            data_entry[SampleType.NAME] = (case_name_xml.text)

            # catchphrases are used as gold standard for summarization
            catchphrases: List[str] = [entry.text for entry in
                                       catchphrases_xml]
            data_entry[SampleType.CATCHPHRASES] = catchphrases

            # sentences are used as input text
            input_text: List[str] = [entry.text for entry in
                                     sentences_xml]
            data_entry[SampleType.INPUT_TEXT] = ' '.join(input_text)

            # check if citation_summ and citation_class files exist,
            # we can add some additional fields here
            if os.path.exists(os.path.join(citation_sum_dir, file)):
                pass
            if os.path.exists(os.path.join(citation_cls_dir, file)):
                pass

            self.data.append(data_entry)
        self.loaded = True

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)


class EnronEmailsTagging(EnronEmails):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: int = 1024, training_ratio=0.7):
        assert max_src_len >= 1024
        super(EnronEmailsTagging, self).__init__(tokenizer, max_src_len,
                                                 256,
                                                 training_ratio=training_ratio)

        self.sample_map: Dict[int, Tuple[int, int]] = {}

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        return len(self.sample_map)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        super()._load(train, val, test)
        self.loaded = False

        # Create a map from sample index to (case index, catchphrase index),
        # this is used if you would like to treat every catchphrase as a
        # separate sample
        total_count = 0
        for i in range(len(self.data)):
            catchphrases: List[str] = self.data[i][SampleType.CATCHPHRASES]
            for j in range(len(catchphrases)):
                self.sample_map[total_count] = (i, j)
                total_count += 1

        # Drop sample types from data that aren't needed
        for i in range(len(self.data)):
            catchphrases: List[str] = self.data[i][SampleType.CATCHPHRASES]
            if len(catchphrases) == 0:
                continue
            self.data[i] = {
                # Input text is joined because we want all sentences, it's a
                # very long text encoding task
                SampleType.INPUT_TEXT: self.data[i][SampleType.INPUT_TEXT],
                SampleType.CATCHPHRASES: catchphrases}
        self.loaded = True

    @overrides(EnronEmails)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        sample_idx, catchphrase_idx = self.sample_map[idx]
        input_text = self.data[sample_idx][SampleType.INPUT_TEXT]
        catchphrase = self.data[catchphrase_idx][SampleType.CATCHPHRASES]
        return input_text, catchphrase


def inspect_catchphrase_diffs():
    # please install HuggingFace ds by pip install ds
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    ds = EnronEmailsTagging(tokenizer=tokenizer,
                            training_ratio=1.0).load_training_data()


def inspect_src_test():
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    ds = EnronEmailsTagging(tokenizer=tokenizer, max_src_len=17000,
                            training_ratio=0.1).load_training_data()

    def print_n_docs(n):
        for i in range(n):
            print('Data Sample {}:'.format(i))
            print(ds.data[n][SampleType.NAME] + '\n')

    print_n_docs(10)


def main():
    inspect_src_test()


if __name__ == '__main__':
    main()
