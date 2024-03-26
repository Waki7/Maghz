from __future__ import annotations

import json
import os
import random
from functools import partial
from pathlib import Path

import torch.utils.data
from bs4 import BeautifulSoup, ResultSet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import SentenceDataset, \
    SampleType, MetaLearningMixIn
from mgz.ds.sentence_datasets.task_creation import \
    tagging_with_semantic_grouping
from mgz.typing import *

DATASET_DIR = os.path.join(
    Path(__file__).resolve().parent.parent.parent.parent,
    'datasets/corpus/').replace("\\", "/")


class LoadingTask:
    '''
    This class is used to load additional categories that were generated programmatically as additional tasks or pretext tasks.
    '''

    def __init__(self):
        pass


class AusCaseReports(SentenceDataset):
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
        super(AusCaseReports, self).__init__(tokenizer=tokenizer,
                                             max_src_len=max_src_len,
                                             max_tgt_len=max_tgt_len,
                                             dataset_dir=dataset_dir)
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
    def input_space(self) -> sp.SentenceQuery:
        return sp.SentenceQuery(
            len((self.vocab_src)), sequence_len=self.max_src_len,
            query_len=self.max_tgt_len)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Sentence:
        return sp.Sentence(len((self.vocab_tgt)),
                           sequence_len=(self.max_tgt_len,))

    def _pre_load_set_state_data_range(self, train: bool,
                                       val: bool,
                                       test: bool) -> range:
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
        # Read all the input text files, from readme:
        # 1. fulltext: Contains the full text and the catchphrases of all the cases from the FCA. Every document (<case>) contains:
        # 	<name> : name of the case
        # 	<AustLII> : link to the austlii page from where the document was taken
        # 	<catchphrases> : contains a list of <catchphrase> elements
        # 		<catchphrase> : a catchphrase for the case, with an id attribute
        # 	<sentences> : contains a list of <sentence> elements
        # 		<sentence> : a sentence with id attribute
        fulltext_dir = os.path.join(DATASET_DIR, 'fulltext')

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
        citation_sum_dir = os.path.join(DATASET_DIR, 'citations_summ')

        # 3. citations_class: Contains for each case a list of labeled citations. Fields:
        # 	<name> : name of the case
        # 	<AustLII> : link to the austlii page from where the document was taken
        # 	<citations> : contains a list of <citation> elements
        # 		<citation> : a citation to an older case, it has an id attribute and contains the following elements:
        # 			<class> : the class of the citation as indicated on the document
        # 			<tocase> : the name of the case which is cited
        # 			<AustLII> : the link to the document of the case which is cited
        # 			<text> : paragraphs in the cited case where the current case is mentioned
        citation_cls_dir = os.path.join(DATASET_DIR, 'citations_class')

        sample_range: range = self._pre_load_set_state_data_range(train, val,
                                                                  test)
        logging.info('Loading data from: ' + fulltext_dir)
        for file in tqdm(
                os.listdir(fulltext_dir)[sample_range[0]: sample_range[-1]]):
            data_entry: Dict[SampleType, Union[str, List[str]]] = \
                {SampleType.KEY: None,
                 SampleType.NAME: None,
                 SampleType.CATCHPHRASES: [],
                 SampleType.INPUT_TEXT: []}

            file_path = os.path.join(fulltext_dir, file)
            with open(file_path) as fp:
                try:
                    soup = BeautifulSoup(fp, 'html.parser')
                except UnicodeDecodeError as e:
                    logging.info(
                        "Error parsing file non utf-8 character found: " + file)
                    continue
                case_name_xml: ResultSet = soup.find('name')
                catchphrases_xml: List[ResultSet] = soup.find_all('catchphrase')
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


class AusCaseReportsToPhraseTag(AusCaseReports):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: int = 1024, training_ratio=0.7,
                 dataset_dir: str = None):
        if dataset_dir is None:
            dataset_dir = DATASET_DIR
        assert max_src_len >= 1024
        super(AusCaseReportsToPhraseTag, self).__init__(tokenizer, max_src_len,
                                                        256,
                                                        training_ratio=training_ratio,
                                                        dataset_dir=dataset_dir)

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

    @overrides(AusCaseReports)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        sample_idx, catchphrase_idx = self.sample_map[idx]
        input_text = self.data[sample_idx][SampleType.INPUT_TEXT]
        catchphrase = self.data[catchphrase_idx][SampleType.CATCHPHRASES]
        return input_text, catchphrase


class AusCaseReportsToTagGrouped(AusCaseReports, MetaLearningMixIn):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 n_episodes: Optional[int], n_queries_per_cls: List[int],
                 n_supports_per_cls: List[int],
                 max_src_len: int = 1024, training_ratio=0.7, max_words_tag=5):
        super(AusCaseReportsToTagGrouped, self).__init__(tokenizer,
                                                         max_src_len=max_src_len,
                                                         max_tgt_len=max_words_tag,
                                                         training_ratio=training_ratio)
        self.cosine_similarity_threshold = 0.94
        self.cosine_threshold_word = 0.88
        self.similar_enough_count_diff = 1
        self.punctuation_to_remove = r"""!? ,.:;?"""
        self.max_words_tag = max_words_tag
        self._tag_to_sample_idx_map: OrderedDict[str, List[int]] = OrderedDict()

        # Meta Learning Sampling Parameters
        self._n_supports_per_cls = n_queries_per_cls
        self._n_supports_per_cls = n_supports_per_cls
        self.n_episodes = n_episodes

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
        assert self.data_state != DataState.NOT_LOADED, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def create_dataloaders(self,
                           device: Union[torch.device, int],
                           batch_size: int = 12000,
                           is_distributed: bool = True,
                           turn_off_shuffle=False,
                           data_sampler: torch.utils.data.Sampler = None
                           ) -> (DataLoader, DataLoader):
        valid_sampler = (
            DistributedSampler(self) if is_distributed else data_sampler
        )
        # TODO we should shuffle ahead of time so that it's consistent,
        #  we can't have none shuffling for this dataset
        shuffle = True  # not turn_off_shuffle and (valid_sampler is None)
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=shuffle,
            sampler=valid_sampler,
            collate_fn=self.get_collate_fn(device)
        )
        return dataloader

    def _get_json_file_name(self):
        # Windows doesn't support special characters in file names, so we
        # convert them to ascii
        ascii_converted_punctuation: str = ''.join(
            str(ord(c)) for c in self.punctuation_to_remove)
        return '-'.join([str(self.data_state), str(
            self.training_ratio), str(
            self.validation_ratio), str(self.test_ratio), str(
            self.max_src_len), str(self.max_words_tag), str(
            self.cosine_similarity_threshold), str(
            self.cosine_threshold_word), str(
            self.similar_enough_count_diff),
                         ascii_converted_punctuation]) + '.json'

    def try_load_from_json(self):
        cache_dir = os.path.join(DATASET_DIR, 'cache')
        # create name from configuration of current class
        processed_data_cache = os.path.join(cache_dir, self.__class__.__name__,
                                            self._get_json_file_name())
        if not os.path.exists(processed_data_cache):
            return False
        with open(processed_data_cache) as file_object:
            # store file data in object
            json_dict: Dict = json.load(file_object)
        json_data = json_dict['data']
        # Deserialize cached data
        for sample in json_data:
            self.data.append({})
            for key, val in sample.items():
                self.data[-1][SampleType(key)] = val
        return True

    def save_tag_grouping_cache(self):
        cache_dir = os.path.join(DATASET_DIR, 'cache')
        if not os.path.exists(cache_dir):
            os.makedirs(cache_dir)
        class_cache_dir = os.path.join(cache_dir, self.__class__.__name__)
        if not os.path.exists(class_cache_dir):
            os.makedirs(class_cache_dir)
        # --- Serialize data to cache ---
        serializable_data = []
        for sample in self.data:
            serializable_data.append({})
            for key, val in sample.items():
                serializable_data[-1][key.value] = val
        # --- Save to json ---
        with open(
                os.path.join(class_cache_dir,
                             self._get_json_file_name()).replace("\\", "/"),
                'w') as f:
            obj_dict = {'data': serializable_data}
            f.write(json.dumps(obj_dict, indent=4, separators=(',', ': ')))

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        self._pre_load_set_state_data_range(train, val, test)
        loaded_from_json = self.try_load_from_json()  # loads self.data

        if not loaded_from_json:
            super()._load(train, val, test)  # Only loads into self.data
            self.loaded = False
            # --- Remove punctuation ---
            if self.punctuation_to_remove:
                for i in range(len(self.data)):
                    for j in range(len(self.data[i][SampleType.CATCHPHRASES])):
                        self.data[i][SampleType.CATCHPHRASES][j] = \
                            self.data[i][SampleType.CATCHPHRASES][j].translate(
                                str.maketrans('', '',
                                              self.punctuation_to_remove))

            # Group tags into semantic groups heuristically
            grouped_tags: List[List[str]] = tagging_with_semantic_grouping(
                [sample[SampleType.CATCHPHRASES] for sample in self.data])
            assert len(grouped_tags) == len(self.data), '{} != {}'.format(
                len(grouped_tags), len(self.data))
            for i in range(len(self.data)):
                self.data[i][SampleType.CATCHPHRASES] = grouped_tags[i]

            new_data: List[Dict] = []
            # ---  Drop sample types from data that aren't needed ---
            for i in range(len(self.data)):
                catchphrases: List[str] = self.data[i][SampleType.CATCHPHRASES]
                if len(catchphrases) == 0 or len(
                        self.data[i][SampleType.INPUT_TEXT]) == 0:
                    logging.info(
                        'Dropping sample with missing catchphrase or input text')
                    continue
                # Input text is joined because we want all sentences, it's a
                # very long text encoding task
                new_data.append({
                    SampleType.INPUT_TEXT: self.data[i][SampleType.INPUT_TEXT],
                    SampleType.CATCHPHRASES: catchphrases})
            self.data = new_data
            cache_dir = os.path.join(DATASET_DIR, 'cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.save_tag_grouping_cache()

        self._tag_to_sample_idx_map: Dict[
            str, List[int]] = self.create_tag_to_sample_idx_map()
        self.loaded = True

    @overrides(AusCaseReports)
    def __getitem__(self, idx) -> (SrcStringT, TgtStringT):
        tags = list(self._tag_to_sample_idx_map.keys())
        tag_idx = idx % len(self._tag_to_sample_idx_map)
        selected_tag: TgtStringT = tags[tag_idx]
        rand_sample_for_tag: SrcStringT = self.data[random.choice(
            self._tag_to_sample_idx_map[selected_tag])][SampleType.INPUT_TEXT]
        return rand_sample_for_tag, selected_tag


def inspect_src_test():
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    ds = AusCaseReports(tokenizer=tokenizer, max_src_len=17000, max_tgt_len=256,
                        training_ratio=0.1).load_training_data()

    def print_n_docs(n):
        for i in range(n):
            print('Data Sample {}:'.format(i))
            print(ds.data[n][SampleType.NAME] + '\n')
            print(ds.data[n][SampleType.INPUT_TEXT] + '\n')
            print(ds.data[n][SampleType.CATCHPHRASES] + '\n')

    print_n_docs(10)


def main():
    inspect_src_test()


if __name__ == '__main__':
    main()
