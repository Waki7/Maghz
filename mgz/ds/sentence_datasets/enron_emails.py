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
from transformers import PreTrainedTokenizer, LlamaTokenizer, LlamaTokenizerFast

import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.gpt_input_augments import tag_question_augment, \
    vague_relevance_augment, summarization_augment
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    Sent2SentBatch, SampleType, MetaLearningMixIn, Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.model_running.run_ops import tag_questions_controller
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
                category_dict: Dict[int, str] = json.load(f)
            categories: List[Tuple[int, ...]] = [
                tuple(map(int, re.findall(r'\d+', label))) for
                label in category_lines]
            tags_for_email: List[str] = \
                [category_dict[c[0]] for c in categories]
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
                        tokenizer: PreTrainedTokenizer) -> int:
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
        return partial(Sent2SentBatch.default_collate_fn, self, device)

    def _pre_load_paths_data_range(self, train: bool,
                                   val: bool,
                                   test: bool) -> Tuple[
        List[FilePath], List[FilePath]]:
        email_file_paths: List[FilePath] = []
        label_file_paths: List[FilePath] = []
        directories = [os.path.join(self.dataset_dir, str(i)) for i in
                       range(1, 9)]
        for dir in directories:
            for file in os.listdir(dir):
                if file.endswith('.txt'):
                    email_file_paths.append(os.path.join(dir, file))
                if file.endswith('.cats'):
                    label_file_paths.append(os.path.join(dir, file))
        email_file_paths = sorted(email_file_paths)
        label_file_paths = sorted(label_file_paths)
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
                 training_ratio: float = 0.75,
                 dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = DATASET_DIR
        max_tgt_len = self.get_max_tgt_len(dataset_dir, tokenizer)
        super(EnronEmailsTagging, self).__init__(tokenizer=tokenizer,
                                                 max_src_len=max_src_len,
                                                 max_tgt_len=max_tgt_len,
                                                 dataset_dir=dataset_dir)
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
        # keys_to_keep = [
        #     SampleType.MESSAGE_ID,
        #     SampleType.DATE,
        #     SampleType.FROM,
        #     SampleType.TO,
        #     SampleType.SUBJECT,
        #     SampleType.MIME_VERSION,
        #     SampleType.CONTENT_TYPE,
        #     SampleType.CONTENT_TRANSFER_ENCODING,
        #     SampleType.X_FROM,
        #     SampleType.X_TO,
        #     SampleType.X_CC,
        #     SampleType.X_BCC,
        #     SampleType.X_FOLDER,
        #     SampleType.X_ORIGIN,
        #     SampleType.X_FILENAME,
        #     # SampleType.FILE_NAME,
        #     SampleType.CATCHPHRASES,
        #     SampleType.FULL_AS_TEXT,
        # ]
        # for i in range(len(self.data)):
        #     doc_filtered = {key: self.data[i][key] for key in keys_to_keep}
        #     self.data[i] = doc_filtered
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
            self._tag_to_sample_idx_map[selected_tag])][
            SampleType.FULL_AS_STRING]
        return rand_sample_for_tag, selected_tag


class EnronEmailsTagQA(EnronEmailsTagging, MetaLearningMixIn):

    def __init__(self, tokenizer: Union[LlamaTokenizerFast, LlamaTokenizer],
                 max_src_len: SrcSeqLen, n_query_per_cls: List[int] = 5,
                 n_support_per_cls: List[int] = 5,
                 n_episodes: int = 100,
                 training_ratio: float = 0.75,
                 dataset_dir: str = None):
        assert isinstance(tokenizer, LlamaTokenizerFast) or isinstance(
            tokenizer, LlamaTokenizer), \
            f'This dataset requires the LLamaTokenizer found {type(tokenizer)}'
        super(EnronEmailsTagQA, self).__init__(tokenizer=tokenizer,
                                               max_src_len=max_src_len,
                                               dataset_dir=dataset_dir,
                                               n_query_per_cls=n_query_per_cls,
                                               n_support_per_cls=n_support_per_cls,
                                               n_episodes=n_episodes,
                                               training_ratio=training_ratio)

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(TagQAMetaTaskBatch.default_collate_fn, self, device)

    @overrides(EnronEmailsTagging)
    def __getitem__(self, idx) -> (SrcStringT, TgtStringT):
        tags = list(self._tag_to_sample_idx_map.keys())
        tag_idx = idx % len(self._tag_to_sample_idx_map)
        selected_tag: TgtStringT = tags[tag_idx]
        rand_sample_for_tag: SrcStringT = self.data[random.choice(
            self._tag_to_sample_idx_map[selected_tag])][
            SampleType.FULL_AS_STRING]
        return rand_sample_for_tag, selected_tag


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


def look_through_data():
    from transformers import LEDTokenizer
    tokenizer = LEDTokenizer.from_pretrained(
        "allenai/primera-multi_lexsum-source-long")
    ds = EnronEmailsTagging(tokenizer,
                            max_src_len=2000,
                            n_episodes=1000).load_training_data()
    for i in ds:
        print(i[0])
        print(i[1])
        print('-----------------')


def dump_n_examples(n: int):
    from mgz.version_control import ModelDatabase, ModelNode

    tag_to_sample = "all documents and communications between enron employees discussing government inquiries and investigations into enron"
    contains_words = ["government", "inquiries", "investigation"]
    p_contained = 0.25
    p_random = 0.09
    model_node: ModelNode = ModelDatabase.mistral_openchat_quantized()
    ds = EnronEmails(model_node.tokenizer,
                     max_src_len=4096, max_tgt_len=-1).load_training_data()
#     answers = tag_questions_controller(
#         model=model_node.model, texts=[r'''Message-ID: <24828229.1075846177387.JavaMail.evans@thyme>
# Date: Wed, 27 Sep 2000 10:33:00 -0700 (PDT)
# From: steven.kean@enron.com
# To: cynthia.sandherr@enron.com
# Subject: Re: Congressman Edwards (D-TX)
# Mime-Version: 1.0
# Content-Type: text/plain; charset=us-ascii
# Content-Transfer-Encoding: 7bit
# X-From: Steven J Kean
# X-To: Cynthia Sandherr
# X-cc:
# X-bcc:
# X-Folder: \Steven_Kean_Dec2000_1\Notes Folders\All documents
# X-Origin: KEAN-S
# X-FileName: skean.nsf
#
# Submit a Pac request and keep Tom posted on when and where.
#
#
#
# 	Cynthia Sandherr
# 	09/07/2000 05:05 PM
#
# 		 To: Steven J Kean/NA/Enron@Enron, Joe Hillings/Corp/Enron@ENRON, Elizabeth
# Linnell/NA/Enron@Enron, Joe Hillings/Corp/Enron@ENRON, Carolyn
# Cooney/Corp/Enron@ENRON
# 		 cc: Rosalee Fleming/Corp/Enron@ENRON, Thomas E White/HOU/EES@EES
# 		 Subject: Congressman Edwards (D-TX)
#
# Steve:  Congressman Chet Edwards is having a fundraiser in Houston, October
# 10th, 5:30-7:30 p.m. at Reliant's Bruce Gibson's home in River Oaks (2178
# Troon).  Drayton McLane will attend.  They are looking for $1000 to host or
# $500 sponsor.  Congressman Edwards has been very helpful with our defense
# procurement issues, and we have contributed to him recently.  The
# Congressman's main request was to include Ken Lay's name on their invitation
# which Ken Lay graciously agreed to do so the Congressman is most pleased.
# Thus, the remaining request to attend and give additional money is up to you,
# although if we have room in our budget and Ken Lay or Tom White were
# available, I would recommend their attendance.
# '''],
#         tags=[tag_to_sample], tokenizer=model_node.tokenizer,
#         augment_function=tag_question_augment,
#         max_new_tokens=100)
#     print(answers)
#     exit(3)
    keys_to_keep = [
        SampleType.MESSAGE_ID,
        SampleType.DATE,
        SampleType.FROM,
        SampleType.TO,
        SampleType.SUBJECT,
        SampleType.MIME_VERSION,
        SampleType.CONTENT_TYPE,
        SampleType.CONTENT_TRANSFER_ENCODING,
        SampleType.X_FROM,
        SampleType.X_TO,
        SampleType.X_CC,
        SampleType.X_BCC,
        SampleType.X_FOLDER,
        SampleType.X_ORIGIN,
        SampleType.X_FILENAME,
        SampleType.FILE_NAME,
        # SampleType.CATCHPHRASES,
        SampleType.BODY,
        SampleType.FULL_AS_STRING,
        # SampleType.FULL_AS_BYTES
    ]
    docs_formatted = []

    export_dir = os.path.join(
        Path(__file__).resolve().parent.parent.parent.parent,
        'datasets/enron_export/').replace("\\", "/")
    if os.path.exists(export_dir):
        shutil.rmtree(export_dir)
    os.mkdir(export_dir)
    os.mkdir(os.path.join(export_dir, 'yes'))
    os.mkdir(os.path.join(export_dir, 'no'))
    os.mkdir(os.path.join(export_dir, 'maybe'))

    bsz = 3
    for i in tqdm(range(0, len(ds), bsz)):
        batch = ds[i:i + bsz]
        doc_batch_filtered = [{key: doc[key] for key
                               in keys_to_keep} for doc in batch]
        full_texts = [doc[SampleType.FULL_AS_STRING] for doc in batch]
        tags: List[str] = [tag_to_sample] * len(full_texts)
        answers = tag_questions_controller(
            model=model_node.model, texts=full_texts,
            tags=tags, tokenizer=model_node.tokenizer,
            augment_function=tag_question_augment,
            max_new_tokens=3)
        for answer, doc in zip(answers, doc_batch_filtered):
            random_p = random.random()
            no_inclusion = "no" in answer.lower() and random_p < p_random
            word_containment_inclusion = any(
                [word in doc[SampleType.FULL_AS_STRING] for word in
                 contains_words]) and random.random() < p_contained
            maybe_inclusion = "maybe" in answer.lower()
            yes_inclusion = "yes" in answer.lower()
            if not (no_inclusion or word_containment_inclusion or
                    maybe_inclusion or yes_inclusion):
                continue
            print('answer', answer)
            docs_formatted.append(
                {str(key) + ".value": doc[key] for key in keys_to_keep})

            if yes_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'yes'))
            elif word_containment_inclusion or maybe_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'maybe'))
            elif no_inclusion:
                shutil.copy(doc[SampleType.FILE_NAME], os.path.join(
                    export_dir, 'no'))

            email_msg: email.message.Message = email.message_from_string(
                doc[SampleType.FULL_AS_STRING])
            assert email_msg.as_string() == doc[SampleType.FULL_AS_STRING]

            # email_msg: email.message.Message = email.message_from_bytes(
            #     doc[SampleType.FULL_AS_BYTES])
            # assert email_msg.as_bytes() == doc[SampleType.FULL_AS_BYTES]
        if len(docs_formatted) > n:
            break

        print(f'collected {len(docs_formatted)} examples so far')

    with open(os.path.join(export_dir, 'tags_sampled.txt'), 'w') as file:
        file.write(tag_to_sample)
    as_dict_file = os.path.join(export_dir, 'formatted_docs.json')

    # Writing to the file
    with open(as_dict_file, 'w') as file:
        file.write('[\n')
        for entry in docs_formatted:
            file.write('\t{\n')
            for key, value in entry.items():
                cleaned_key = key.replace('\"', '')
                file.write(f"\t\t{cleaned_key}: {repr(value)},\n")
            file.write('\t},\n')
        file.write(']\n')

    # Reading from the file and printing its contents
    with open(as_dict_file, 'r') as file:
        content = file.read()
        print(content)


class CustomTqdm(tqdm):
    @staticmethod
    def get_for_job(job_id: int):
        return partial(CustomTqdm, job_id=job_id)

    def __init__(self, iterable, desc=None, leave=True, file=None,
                 job_id: int = None):
        super().__init__(iterable, desc, leave, file, mininterval=2.0)
        self.job_id = job_id

    @overrides(tqdm)
    def update(self, n=30):
        super().update(n)
        print('total', len(self.iterable))
        print('update', n)
        print(self.job_id)


def main():
    dump_n_examples(30)


if __name__ == '__main__':
    main()
