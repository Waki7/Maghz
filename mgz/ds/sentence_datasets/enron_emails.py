from __future__ import annotations

import email
import os
import random
import re
from functools import partial
from pathlib import Path

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    Sent2SentBatch, SampleType, MetaLearningMixIn, Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.typing import *

DATASET_DIR = os.path.join(
    Path(__file__).resolve().parent.parent.parent.parent,
    'datasets/enron_with_categories/').replace("\\", "/")
CATEGORIES = {
    1: {
        1: "Company Business or strategy",
        2: "Purely Personal",
        3: "Personal but in professional context",
        4: "Logistic Arrangements",
        5: "Employment arrangements",
        6: "Document editing or checking collaboration",
        7: "missing attachment",
        8: "Empty message",
    },
    2: {
        1: "Includes new text in addition to forwarded material",
        2: "Forwarded emails including replies",
        3: "Business letters or documents",
        4: "News articles",
        5: "Government or academic reports",
        6: "Government actions",
        7: "Press releases",
        8: "Legal documents",
        9: "Pointers to urls",
        10: "Newsletters",
        11: "Jokes, humor related to business",
        12: "Jokes, humor unrelated to business",
        13: "Attachments",
    },
    3: {
        1: "regulations and regulators",
        2: "internal projects",
        3: "current company image",
        4: "changing company image",
        5: "political influence or contributions or contacts",
        6: "california energy crisis or politics",
        7: "company policy",
        8: "company operations",
        9: "alliances or partnerships",
        10: "legal advice",
        11: "talking points",
        12: "meeting minutes",
        13: "trip reports",
    },
    4: {
        1: "jubilation",
        2: "hope or anticipation",
        3: "humor",
        4: "camaraderie",
        5: "admiration",
        6: "gratitude",
        7: "friendship or affection",
        8: "sympathy or support",
        9: "sarcasm",
        10: "secrecy or confidentiality",
        11: "worry or anxiety",
        12: "concern",
        13: "competitiveness or aggressiveness",
        14: "triumph or gloating",
        15: "pride",
        16: "anger or agitation",
        17: "sadness or despair",
        18: "shame",
        19: "dislike or scorn",
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
        self.validation_ratio = .2
        self.test_ratio = .1

    @overrides(BaseDataset)
    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        logging.info('Reading data from: ' + DATASET_DIR)
        email_file_paths, label_file_paths = self._pre_load_paths_data_range(
            train, val, test)

        for email_path, label_path in tqdm(
                zip(email_file_paths, label_file_paths)):
            with open(label_path) as f:
                # label shows up as 'int,int,int\n'
                labels: List[str] = f.readlines()
                categories: List[Tuple[int, ...]] = [
                    tuple(map(int, re.findall(r'\d+', label))) for
                    label in labels]
                tags_for_email: List[str] = \
                    [CATEGORIES[c[0]][c[1]] for c in categories]

            # Read txt file formatted as e-mail with library email
            emails: email.message.Message = email.message_from_file(
                open(email_path))

            data_entry: Dict[SampleType, Union[str, List[str]]] = \
                {SampleType.KEY: emails.get('Message-ID'),
                 SampleType.NAME: emails.get('Subject'),
                 SampleType.CATCHPHRASES: tags_for_email,
                 SampleType.INPUT_TEXT: '\n'.join([emails.get('Subject'),
                                                   emails.get_payload()])}
            self.data.append(data_entry)
        self.loaded = True

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
    def input_space(self) -> sp.SentenceQuery:
        return sp.SentenceQuery(
            len((self.vocab_src)), sequence_len=self.max_src_len,
            query_len=self.max_tgt_len)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Sentence:
        return sp.Sentence(len((self.vocab_tgt)),
                           sequence_len=self.max_tgt_len)

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2SentBatch.default_collate_fn, self, device)

    def _pre_load_paths_data_range(self, train: bool,
                                   val: bool,
                                   test: bool) -> Tuple[
        List[FilePath], List[FilePath]]:
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
        n_files = len(email_file_paths)
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
        return email_file_paths[sample_range[0]: sample_range[-1]], \
            label_file_paths[sample_range[0]: sample_range[-1]]

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)


class EnronEmailsTagging(EnronEmails, MetaLearningMixIn):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: SrcSeqLen, n_query_per_cls: List[int] = 5,
                 n_support_per_cls: List[int] = 5,
                 n_episodes: int = 100,
                 training_ratio: float = 0.75):
        all_tags = []
        [all_tags.extend(subcategories.values()) for subcategories in
         CATEGORIES.values()]
        max_tgt_len = max([len(tokenizer.tokenize(tag)) for tag in all_tags])
        super(EnronEmails, self).__init__(tokenizer=tokenizer,
                                          max_src_len=max_src_len,
                                          max_tgt_len=max_tgt_len)
        # --- Initialization flags ---
        self.training_ratio = training_ratio
        self.validation_ratio = .2
        self.test_ratio = .1

        self._tag_to_sample_idx_map: OrderedDict[str, List[int]] = OrderedDict()

        # Meta Learning Sampling Parameters
        self._n_query_per_cls = n_query_per_cls  # Will be roughly n_shot per class, not exact
        self._n_support_per_cls = n_support_per_cls  # Will be roughly n_shot per class, not exact
        self.n_episodes = n_episodes

    @property
    @overrides(SentenceDataset)
    def input_space(self) -> sp.Sentence:
        return sp.Sentence(
            len((self.vocab_src)), sequence_len=self.max_src_len)

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        super()._load(train, val, test)  # Only loads into self.data
        self.loaded = False
        self._tag_to_sample_idx_map: Dict[
            str, List[int]] = self.create_tag_to_sample_idx_map()
        self.loaded = True

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        if self.n_episodes is not None:
            return self.n_episodes
        return len(self._tag_to_sample_idx_map)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Tagging:
        return sp.BinaryTagging()

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(Sent2TagMetaTaskBatch.default_collate_fn, self, device)

    @overrides(EnronEmails)
    def __getitem__(self, idx) -> (SrcStringT, TgtStringT):
        tags = list(self._tag_to_sample_idx_map.keys())
        tag_idx = idx % len(self._tag_to_sample_idx_map)
        selected_tag: TgtStringT = tags[tag_idx]
        rand_sample_for_tag: SrcStringT = self.data[random.choice(
            self._tag_to_sample_idx_map[selected_tag])][SampleType.INPUT_TEXT]
        return rand_sample_for_tag, selected_tag


class EnronEmailsTagQA(EnronEmailsTagging, MetaLearningMixIn):
    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(TagQAMetaTaskBatch.default_collate_fn, self, device)


def inspect_catchphrase_diffs():
    # please install HuggingFace ds by pip install ds
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    ds = EnronEmailsTagging(tokenizer=tokenizer,
                            training_ratio=1.0).load_training_data()


def inspect_src_test():
    from transformers import LEDTokenizer
    tokenizer = LEDTokenizer.from_pretrained(
        "allenai/primera-multi_lexsum-source-long")
    ds = EnronEmailsTagging(tokenizer,
                            max_src_len=2000,
                            n_episodes=1000,
                            task_size_per_cls=2).load_training_data()
    max_lens = []
    max_emails = []
    i: Tuple[str, str]
    for i in ds:
        lent = len(tokenizer.tokenize(i[0]))
        email = i[0]
        if len(max_emails) < 10:
            max_emails.append(email)
            max_lens.append(lent)
        else:
            lowest_high_val = np.argmin(max_lens)
            if lent > max_lens[lowest_high_val]:
                max_lens.pop(lowest_high_val)
                max_emails.pop(lowest_high_val)
                max_lens.append(lent)
                max_emails.append(email)
    print(max_lens)


def main():
    inspect_src_test()


if __name__ == '__main__':
    main()
