from __future__ import annotations

import json
import os
import random
from collections import defaultdict
from functools import partial
from pathlib import Path

import torch.utils.data
from bs4 import BeautifulSoup, ResultSet
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import mgz.models.nlp.metrics as metrics
import spaces as sp
from mgz.ds.base_dataset import BaseDataset, DataState
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    Sent2TagMetaTaskBatch, Sent2SentBatch, SampleType
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
                 training_ratio=0.7):  # change for testing/verification
        super(AusCaseReports, self).__init__(tokenizer=tokenizer,
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
                data_entry[SampleType.INPUT_TEXT] = input_text

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
                 max_position_embeddings: int, training_ratio=0.7):
        assert max_position_embeddings >= 1024
        super(AusCaseReportsToPhraseTag, self).__init__(tokenizer, 1024,
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
                SampleType.INPUT_TEXT: self.data[i][SampleType.INPUT_TEXT],
                SampleType.CATCHPHRASES: catchphrases}
        self.loaded = True

    @overrides(AusCaseReports)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        sample_idx, catchphrase_idx = self.sample_map[idx]
        input_text = self.data[sample_idx][SampleType.INPUT_TEXT]
        catchphrase = self.data[catchphrase_idx][SampleType.CATCHPHRASES]
        return input_text, catchphrase


class AusCaseReportsToTagGrouped(AusCaseReports):
    def __init__(self, tokenizer: PreTrainedTokenizer,
                 max_src_len: int, training_ratio=0.7, n_shot=5,
                 max_words_tag=5):
        assert max_src_len >= 1024

        super(AusCaseReportsToTagGrouped, self).__init__(tokenizer,
                                                         max_src_len=1024,
                                                         max_tgt_len=max_words_tag,
                                                         training_ratio=training_ratio)
        self.cosine_similarity_threshold = 0.94
        self.cosine_threshold_word = 0.88
        self.similar_enough_count_diff = 1
        self.punctuation_to_remove = r"""!?,.:;?"""
        self.max_words_tag = max_words_tag
        self.tag_to_sample_idx_map: Dict[str, List[int]] = {}

        # Meta Learning Sampling Parameters
        self.n_shot = n_shot  # Will be roughly n_shot, not exact

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        return len(self.data)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Tagging:
        return sp.BinaryTagging()

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[SrcStringT, List[TgtStringT]]]):
        assert len(batch) == 1, "Batch size must be 1 for meta-learning for now"
        # Examples that do and don't have the tag respectively
        positive_examples: set[int] = set()
        negative_examples: set[int] = set()

        # For now we are going to randomly sample a tag and use that, but in the future I think we want to handle that differently
        n_tgt = 1
        pos_tags = [random.sample(sample[1], n_tgt)[0] for sample in batch]
        for pos_tag in pos_tags:
            # Based on the batch, we want to sample additional catchphrases to create a meta learning task out of the batch
            for sample_text, pos_tags in batch:
                timeout = 0  # catch when we can't find a negative example or other positive examples
                while len(
                        positive_examples) < self.n_shot and timeout < 2 * self.n_shot:
                    if len(positive_examples) < self.n_shot:
                        pos_sample_idxs: List[int] = self.tag_to_sample_idx_map[
                            pos_tag]
                        if len(pos_sample_idxs) < self.n_shot:
                            logging.warning(
                                'Not enough samples with tag \'%s\' and training shot of %s, only %s present' % (
                                    pos_tag, self.n_shot, len(pos_sample_idxs)))
                            if self.data_state == DataState.TRAIN:
                                # We may want to skip this batch in the future
                                logging.error(
                                    'Not enough samples with tag \'%s\' and training shot of %s, only %s present' % (
                                        pos_tag, self.n_shot,
                                        len(pos_sample_idxs)))

                        pos_sample_idx = random.choice(pos_sample_idxs)
                        positive_examples.add(pos_sample_idx)

                    # TODO: experiment with pulling negative samples that have very
                    #  different embeddings
                    if len(negative_examples) < self.n_shot:
                        neg_sample_idx = random.randint(0, len(self.data) - 1)
                        neg_tags = self.data[neg_sample_idx][
                            SampleType.CATCHPHRASES]
                        if pos_tag not in neg_tags:
                            negative_examples.add(neg_sample_idx)
                    timeout += 1
        assert self.data_state != DataState.NOT_LOADED, "Dataset not loaded"

        pos_batch: List[SrcStringT] = [self.data[i][SampleType.INPUT_TEXT] for i
                                       in positive_examples]

        neg_srcs: List[SrcStringT] = [self.data[i][SampleType.INPUT_TEXT]
                                      for i in negative_examples]

        tag_task: Dict[
            TgtStringT, Tuple[List[SrcStringT], List[SrcStringT]]] = {
            pos_tag: (pos_batch, neg_srcs)
        }

        def tokenize_src(src_text: SrcTextT) -> List[str]:
            return self.tokenizer_tgt.tokenize(src_text)

        def tokenize_tgt(text: TgtTextT) -> List[str]:
            return self.tokenizer_tgt.tokenize(text)

        def vocab_src(tokens: List[str]) -> List[int]:
            return [self.vocab_src[token] for token in tokens]

        def vocab_tgt(tokens: List[str]) -> List[int]:
            return [self.vocab_tgt[token] for token in tokens]

        return Sent2TagMetaTaskBatch.collate_batch(
            tag_task=tag_task,
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
        dataloader = DataLoader(
            self,
            batch_size=batch_size,
            shuffle=not turn_off_shuffle and (valid_sampler is None),
            sampler=valid_sampler,
            collate_fn=self.get_collate_fn(device)
        )
        return dataloader

    def _get_json_file_name(self):
        return str(self.data_state) + '-' + str(
            self.training_ratio) + '_' + str(
            self.validation_ratio) + '_' + str(self.test_ratio) + '_' + str(
            self.max_src_len) + '_' + str(self.max_words_tag) + '_' + str(
            self.cosine_similarity_threshold) + '_' + str(
            self.cosine_threshold_word) + '_' + str(
            self.similar_enough_count_diff) + str(
            self.punctuation_to_remove) + '.json'

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

    def save_to_json(self):
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
            print(len(self.data))
            print(self.data_state)
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

            # Drop sample types from data that aren't needed
            for i in range(len(self.data)):
                catchphrases: List[str] = self.data[i][SampleType.CATCHPHRASES]
                # Input text is joined because we want all sentences, it's a very long text encoding task
                self.data[i] = {
                    SampleType.INPUT_TEXT: ' '.join(
                        self.data[i][SampleType.INPUT_TEXT]),
                    SampleType.CATCHPHRASES: catchphrases}

            cache_dir = os.path.join(DATASET_DIR, 'cache')
            if not os.path.exists(cache_dir):
                os.makedirs(cache_dir)
            self.save_to_json()

        # Map from tag to sample idx, to know which samples each tag apply to
        for i in range(len(self.data)):
            for catchphrase in self.data[i][SampleType.CATCHPHRASES]:
                if catchphrase not in self.tag_to_sample_idx_map:
                    self.tag_to_sample_idx_map[catchphrase] = []
                self.tag_to_sample_idx_map[catchphrase].append(i)
        self.loaded = True

    @overrides(AusCaseReports)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        input_text = self.data[idx][SampleType.INPUT_TEXT]
        catchphrase = self.data[idx][SampleType.CATCHPHRASES]
        return input_text, catchphrase


class PhrasePairDifference:
    def __init__(self, phrase1: str, phrase2: str,
                 tokenizer: PreTrainedTokenizer,
                 word_diff: int = None,
                 bleu_score: float = None, cosine_similarity: float = None):
        ''' phrase1 and phrase2 are lists of words, setting word_diff and
        bleu_score are completely optional, they will be computed if not set'''
        self.phrase1 = phrase1
        self.phrase2 = phrase2
        self.phrase1_tokenized = tokenizer.tokenize(phrase1)
        self.phrase2_tokenized = tokenizer.tokenize(phrase2)
        self.word_diff: int = metrics.word_count_diff(self.phrase1_tokenized,
                                                      self.phrase2_tokenized) if word_diff is None else word_diff
        self.bleu_score: float = bleu_score
        self.cosine_similarity: float = cosine_similarity

    def _load_scores(self):
        if self.bleu_score is None:
            self.bleu_score = \
                metrics.bleu_from_tokenized_sentences(self.phrase1_tokenized,
                                                      self.phrase2_tokenized,
                                                      max_n=2)
        if self.cosine_similarity is None:
            self.cosine_similarity = \
                metrics.cosine_similarity_from_raw_sentences(self.phrase1,
                                                             self.phrase2)[0]

    def __getitem__(self, item) -> List[str]:
        if item == 0:
            return self.phrase1
        elif item == 1:
            return self.phrase2
        else:
            raise IndexError("PhrasePairDifference only has two phrases")

    def n_total_tokens(self):
        return len(self.phrase1_tokenized) + len(self.phrase2_tokenized)

    def print_diff(self):
        self._load_scores()
        print('----\n\t',
              "Phrase: \n\t\t {} ...tokenized as: {}  \n\t differs from: \n\t\t {} ...tokenized as: {} \n\t by "
              "{} words with a score of {} and cosine similarity of {}".format(
                  self.phrase1, self.phrase1_tokenized, self.phrase2,
                  self.phrase2_tokenized, self.word_diff, self.bleu_score,
                  self.cosine_similarity))


def investigate_catchphrase_differences(ds: AusCaseReports,
                                        tokenizer: PreTrainedTokenizer):
    # We are arbitrarily looking at the first sample as a reference
    catchphrases = ds.data[0][SampleType.CATCHPHRASES]
    total = 0
    phrase_pair_diff_count: Dict[int, int] = defaultdict(lambda: 0)
    phrase_pair_differences: Dict[
        int, List[PhrasePairDifference]] = defaultdict(
        lambda: [])
    # --- Go through every other sample and for each sample, get the most
    # similar catchphrase pair between the reference sample and the sample
    # being searched over. ---
    for idx, d in enumerate(ds.data):
        other_catchphrases = d[SampleType.CATCHPHRASES]
        if len(other_catchphrases) == 0:
            continue
        for phrase in catchphrases:
            tokenized_phrase = tokenizer.tokenize(phrase)
            closest_diff = float('inf')
            phrase_pair_closest: Tuple[str, str] = None
            for other_catchphrase in other_catchphrases:
                tokenized_other_catchphrase = tokenizer.tokenize(
                    other_catchphrase)
                word_diff = metrics.word_count_diff(tokenized_phrase,
                                                    tokenized_other_catchphrase)
                if closest_diff > word_diff:
                    phrase_pair_closest = (phrase, other_catchphrase)
                    closest_diff = word_diff
            total += 1
            phrase_pair_diff_count[closest_diff] += 1
            phrase_pair_differences[closest_diff].append(
                PhrasePairDifference(phrase_pair_closest[0],
                                     phrase_pair_closest[1],
                                     tokenizer, closest_diff))

    print(
        'Analysis of catchphrase differences. We are using just first '
        'sample\'s catchphrases as reference. We take each of the first '
        'samples catchphrase and comapre it to every other catchphrase, '
        'this is how often the exact same catch phrase is used or a small '
        'deviation, measured as how many words differ.')
    for diff, count in phrase_pair_diff_count.items():
        print(
            'Number of catchphrases with {} word difference: {}'.format(diff,
                                                                        count))
        print('multi word phrases that differ by <3 words: ')
        if diff < 3:
            for phrase in phrase_pair_differences[diff]:
                if phrase.n_total_tokens() >= 3:
                    phrase.print_diff()


def main():
    # please install HuggingFace ds by pip install ds
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    ds = AusCaseReportsToTagGrouped(tokenizer=tokenizer,
                                    max_src_len=10000,
                                    training_ratio=1.0).load_training_data()
    investigate_catchphrase_differences(ds, tokenizer)


if __name__ == '__main__':
    main()
