from __future__ import annotations

import email
import json
import os
import random
import re
import shutil
from functools import partial
from pathlib import Path

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizerBase, LlamaTokenizer

import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.gpt_input_augments import PromptConfig
from mgz.ds.sentence_datasets.heuristic_matching import DocumentRuleEvaluator
from mgz.ds.sentence_datasets.responsivenes_datasets.responsive_batch import \
    TagQAMetaTaskBatch
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
     SampleType, MetaLearningMixIn
from mgz.model_running.run_ops import prompt_lm_logits_controller
from mgz.models.nlp.base_transformer import InferenceContext
from mgz.typing import *

DATASET_DIR = os.path.join(
    Path(__file__).resolve().parent.parent.parent.parent,
    'datasets/enron_with_categories/').replace("\\", "/")
ENRON_FALLBACK_CATEGORIES: Dict[int, Dict[int, str]] = {
    1: {
        1: "Coarse genre: Company Business, Strategy, etc.",
        2: "Coarse genre: Purely Personal",
        3: "Coarse genre: Personal but in professional context (e.g., it was good working with you)",
        4: "Coarse genre: Logistic Arrangements (meeting scheduling, technical support, etc)",
        5: "Coarse genre: Employment arrangements (job seeking, hiring, recommendations, etc)",
        6: "Coarse genre: Document editing/checking (collaboration)",
        7: "Coarse genre: Empty message (due to missing attachment)",
        8: "Coarse genre: Empty message",
    },
    2: {
        1: "Included/forwarded information: Includes new text in addition to forwarded material",
        2: "Included/forwarded information: Forwarded email(s) including replies",
        3: "Included/forwarded information: Business letter(s) / document(s)",
        4: "Included/forwarded information: News article(s)",
        5: "Included/forwarded information: Government / academic report(s)",
        6: "Included/forwarded information: Government action(s) (such as results of a hearing, etc)",
        7: "Included/forwarded information: Press release(s)",
        8: "Included/forwarded information: Legal documents (complaints, lawsuits, advice)",
        9: "Included/forwarded information: Pointers to url(s)",
        10: "Included/forwarded information: Newsletters",
        11: "Included/forwarded information: Jokes, humor (related to business)",
        12: "Included/forwarded information: Jokes, humor (unrelated to business)",
        13: "Included/forwarded information: Attachment(s) (assumed missing)",
    },
    3: {
        1: "Primary topics: regulations and regulators (includes price caps)",
        2: "Primary topics: internal projects -- progress and strategy",
        3: "Primary topics: company image -- current",
        4: "Primary topics: company image -- changing / influencing",
        5: "Primary topics: political influence / contributions / contacts",
        6: "Primary topics: california energy crisis / california politics",
        7: "Primary topics: internal company policy",
        8: "Primary topics: internal company operations",
        9: "Primary topics: alliances / partnerships",
        10: "Primary topics: legal advice",
        11: "Primary topics: talking points",
        12: "Primary topics: meeting minutes",
        13: "Primary topics: trip reports",
    },
    4: {
        1: "Emotional tone: jubilation",
        2: "Emotional tone: hope / anticipation",
        3: "Emotional tone: humor",
        4: "Emotional tone: camaraderie",
        5: "Emotional tone: admiration",
        6: "Emotional tone: gratitude",
        7: "Emotional tone: friendship / affection",
        8: "Emotional tone: sympathy / support",
        9: "Emotional tone: sarcasm",
        10: "Emotional tone: secrecy / confidentiality",
        11: "Emotional tone: worry / anxiety",
        12: "Emotional tone: concern",
        13: "Emotional tone: competitiveness / aggressiveness",
        14: "Emotional tone: triumph / gloating",
        15: "Emotional tone: pride",
        16: "Emotional tone: anger / agitation",
        17: "Emotional tone: sadness / despair",
        18: "Emotional tone: shame",
        19: "Emotional tone: dislike / scorn",
    },
}


class EnronEmails(SentenceDataset):
    '''
    Loading pattern for this class is that the parent will load all the data
    and the children will load a subset of a modified subset of the data.
    There are children that are also meta learning datasets.
    '''

    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: Optional[TgtSeqLen] = None,
                 training_ratio=0.7,
                 dataset_dir: str = None):  # change for testing/verification
        if dataset_dir is None:
            dataset_dir = DATASET_DIR
        super(EnronEmails, self).__init__(tokenizer=tokenizer,
                                          max_src_len=max_src_len,
                                          max_tgt_len=max_tgt_len,
                                          dataset_dir=dataset_dir)
        # --- Initialization flags ---
        self.training_ratio = training_ratio
        self.validation_ratio = .2
        self.test_ratio = .1

    @staticmethod
    def get_category_names(category_lines: List[str], dataset_dir: DirPath) -> \
            List[str]:
        # If the dataset is in a generated labeled data format
        if os.path.exists(os.path.join(dataset_dir, 'categories_map.json')):
            with open(os.path.join(dataset_dir, 'categories_map.json')) as f:
                category_dict: Dict[str, str] = json.load(f)
            categories: List[Tuple[int, ...]] = [
                tuple(map(int, re.findall(r'\d+', label))) for
                label in category_lines]
            tags_for_email: List[str] = \
                [category_dict[str(c[0])] for c in categories]
            return tags_for_email
        # If the dataset is in the original enron format
        else:
            categories: List[Tuple[int, ...]] = [
                tuple(map(int, re.findall(r'\d+', label))) for
                label in category_lines]
            tags_for_email: List[str] = \
                [ENRON_FALLBACK_CATEGORIES[c[0]][c[1]] for c in categories]
            return tags_for_email

    @staticmethod
    def get_max_tgt_len(dataset_dir: DirPath,
                        tokenizer: PreTrainedTokenizerBase) -> int:
        all_tags: List[str] = []
        # If the dataset is in a generated labeled data format
        if os.path.exists(os.path.join(dataset_dir, 'categories_map.json')):
            with open(os.path.join(dataset_dir, 'categories_map.json')) as f:
                category_dict: Dict[int, str] = json.load(f)
            all_tags = list(category_dict.values())
        # If the dataset is in the original enron format
        else:
            categories: Dict[int, Dict[int, str]] = ENRON_FALLBACK_CATEGORIES
            [all_tags.extend(subcategories.values()) for subcategories in
             categories.values()]
        max_tgt_len = max([len(tokenizer.tokenize(tag)) for tag in all_tags])
        return max_tgt_len

    @overrides(BaseDataset)
    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        logging.info('Reading data from: ' + self.dataset_dir)
        email_file_paths, label_file_paths = self._pre_load_paths_data_range(
            train, val, test)

        for email_path, label_path in tqdm(
                zip(email_file_paths, label_file_paths)):
            with open(label_path) as f:
                # label shows up as 'int,int,int\n'
                labels: List[str] = f.readlines()
                tags_for_email: List[str] = self.get_category_names(labels,
                                                                    self.dataset_dir)

            # Read txt file formatted as e-mail with library email
            email_msg: email.message.Message = email.message_from_file(
                open(email_path))
            data_entry: Dict[SampleType, Union[str, List[str]]] = \
                {
                    SampleType.MESSAGE_ID: email_msg.get('Message-ID'),
                    SampleType.DATE: email_msg.get('Date'),
                    SampleType.FROM: email_msg.get('From'),
                    SampleType.TO: email_msg.get('To'),
                    SampleType.SUBJECT: email_msg.get('Subject'),
                    SampleType.MIME_VERSION: email_msg.get('Mime-Version'),
                    SampleType.CONTENT_TYPE: email_msg.get('Content-Type'),
                    SampleType.CONTENT_TRANSFER_ENCODING: email_msg.get(
                        'Content-Transfer-Encoding'),
                    SampleType.X_FROM: email_msg.get('X-From'),
                    SampleType.X_TO: email_msg.get('X-To'),
                    SampleType.X_CC: email_msg.get('X-cc'),
                    SampleType.X_BCC: email_msg.get('X-bcc'),
                    SampleType.X_FOLDER: email_msg.get('X-Folder'),
                    SampleType.X_ORIGIN: email_msg.get('X-Origin'),
                    SampleType.X_FILENAME: email_msg.get('X-FileName'),
                    SampleType.FILE_NAME: email_path,
                    SampleType.CATCHPHRASES: tags_for_email,
                    SampleType.BODY: email_msg.get_payload(),
                    SampleType.FULL_AS_STRING: email_msg.as_string(),
                    SampleType.FULL_AS_BYTES: email_msg.as_bytes()
                }
            self.data.append(data_entry)
        self.loaded = True

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        return len(self.data)

    def __getitem__(self, idx) -> Dict:
        return self.data[idx]

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
        # return partial(Sent2SentBatch.default_collate_fn, self, device)
        raise NotImplementedError

    def _pre_load_paths_data_range(self, train: bool,
                                   val: bool,
                                   test: bool) -> Tuple[
        List[FilePath], List[FilePath]]:
        email_file_paths: List[FilePath] = []
        label_file_paths: List[FilePath] = []

        for root, directories, files in os.walk(self.dataset_dir):
            for directory in directories:
                dir_path: DirPath = os.path.join(root, directory)
                for file in os.listdir(dir_path):
                    if file.endswith('.txt'):
                        email_file_paths.append(os.path.join(dir_path, file))
                    if file.endswith('.cats'):
                        label_file_paths.append(os.path.join(dir_path, file))
        email_file_paths = sorted(email_file_paths)
        label_file_paths = sorted(label_file_paths)
        if len(label_file_paths) == 0:
            for email_file_path in email_file_paths:
                parent_dir = os.path.split(os.path.dirname(email_file_path))[-1]
                assert "no" in parent_dir or "maybe" in parent_dir or "yes" in parent_dir, \
                    f"This code is filling in labels from the directory name, it only expects this configuration atm, not {parent_dir}"
                label_str = ""
                if "yes" in email_file_path:
                    label_str = "1"
                label_path = email_file_path.replace('.txt', '.cats')
                label_file_paths.append(label_path)
                with open(label_path, 'w') as f:
                    f.write(label_str)
        assert len(email_file_paths) == len(
            label_file_paths), f'Email files mismatch number of label files'
        assert len(email_file_paths) > 0, f'No email files found'
        idxs = list(range(len(email_file_paths)))
        random.shuffle(idxs)
        email_file_paths = [email_file_paths[i] for i in idxs]
        label_file_paths = [label_file_paths[i] for i in idxs]

        n_files = len(email_file_paths)
        n_train = int(n_files * self.training_ratio)
        n_val = int(n_files * self.validation_ratio)
        n_test = n_files - n_train - n_val
        if not any([train, val, test]):
            raise ValueError("No dataset split selected")

        sample_range = (0, n_files)
        if train:
            sample_range = (0, n_train)
            self.data_state = DataState.TRAIN
        elif val:
            sample_range = (n_train, n_train + n_val)
            self.data_state = DataState.VAL
        elif test:
            sample_range = (n_files - n_test, n_files)
            self.data_state = DataState.TEST

        return email_file_paths[sample_range[0]: sample_range[-1]], \
            label_file_paths[sample_range[0]: sample_range[-1]]

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)
