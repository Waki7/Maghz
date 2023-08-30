from __future__ import annotations

import os
from collections import defaultdict
from functools import partial

from bs4 import BeautifulSoup, ResultSet
from transformers import PreTrainedTokenizerBase

import spaces as sp
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    collate_batch, SampleType
from mgz.models.nlp.metrics import word_count_diff, \
    bleu_from_tokenized_sentences, cosine_similarity_from_raw_sentences
from mgz.models.nlp.tokenizing import TokenStrings
from mgz.typing import *

DATASET_DIR = '../../../../datasets/corpus'


class AusCaseReports(SentenceDataset):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 max_src_len: SrcSeqLen,
                 max_tgt_len: TgtSeqLen):
        super(AusCaseReports, self).__init__()

        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len

        # {'INPUT_TEXT': List[str], 'CATCHPHRASES': List[str],
        # 'summary/short': str, 'summary/tiny': str, 'id': str}
        self.data: List[Dict[SampleType, Union[
            SummaryT, List[SummaryT], SrcTextT, List[SrcTextT]]]] = []

        self.tokenizer_src: PreTrainedTokenizerBase = tokenizer
        self.tokenizer_tgt: PreTrainedTokenizerBase = tokenizer
        self.vocab_src: Dict[str, int] = self.tokenizer_src.get_vocab()
        self.vocab_tgt: Dict[str, int] = self.tokenizer_tgt.get_vocab()

        # --- Initialization flags ---
        self.training_ratio = .7
        self.validation_ratio = .15
        self.test_ratio = .15

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
    def input_space(self) -> sp.SentenceT:
        return sp.SentenceT(
            len((self.vocab_src)), shape=(self.max_src_len,))

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> Union[sp.SentenceT, sp.RegressionTarget]:
        return sp.SentenceT(len((self.vocab_tgt)),
                            shape=(self.max_tgt_len,))

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

        total_samples = len(os.listdir(fulltext_dir))
        n_train = (int)(total_samples * self.training_ratio)
        n_val = (int)(total_samples * self.validation_ratio)
        n_test = (int)(total_samples * self.test_ratio)
        sample_range: Tuple[int, int] = None
        if train:
            sample_range = range(0, n_train)
        elif val:
            sample_range = range(n_train, n_train + n_val)
        elif test:
            sample_range = range(total_samples - n_test, total_samples)
        if sample_range is None:
            raise ValueError("No dataset split selected")

        # src_type: SampleType
        # tgt_type: SampleType
        # src_type, tgt_type = self._get_source_types()

        for file in os.listdir(fulltext_dir)[sample_range[0]: sample_range[-1]]:
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
                if os.path.exists(os.path.join(citation_sum_dir, file)):
                    pass

                self.data.append(data_entry)
            self.loaded = True

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

    def pad_idx(self) -> int:
        return self.vocab_tgt['<blank>']

    def src_vocab_len(self) -> int:
        return len(self.vocab_src)

    def tgt_vocab_len(self) -> int:
        return len(self.vocab_tgt)


class AusCaseReportsToTag(AusCaseReports):
    def __init__(self, tokenizer: PreTrainedTokenizerBase,
                 max_position_embeddings: int):
        assert max_position_embeddings >= 1024
        super(AusCaseReportsToTag, self).__init__(tokenizer, 1024,
                                                  256)

    def _get_source_types(self) -> Tuple[SampleType, SampleType]:
        return SampleType.SUMMARY_LONG, SampleType.SUMMARY_TINY

    @overrides(AusCaseReports)
    def __getitem__(self, idx) -> (SourceListT, SummaryT):
        entry = self.data[idx]
        return entry[SampleType.INPUT_TEXT.value], entry[
            SampleType.CATCHPHRASES.value]


class PhrasePairDifference:
    def __init__(self, phrase1: List[str], phrase2: List[str],
                 tokenizer: PreTrainedTokenizerBase,
                 word_diff: int = None,
                 bleu_score: float = None, cosine_similarity: float = None):
        ''' phrase1 and phrase2 are lists of words, setting word_diff and
        bleu_score are completely optional, they will be computed if not set'''
        self.phrase1 = phrase1
        self.phrase2 = phrase2
        self.phrase1_tokenized = tokenizer.tokenize(phrase1)
        self.phrase2_tokenized = tokenizer.tokenize(phrase2)
        self.word_diff: int = word_count_diff(phrase1,
                                              phrase2) if word_diff is None else word_diff
        self.bleu_score: float = bleu_score
        self.cosine_similarity: float = cosine_similarity

    def _load_scores(self):
        if self.bleu_score is None:
            self.bleu_score = \
                bleu_from_tokenized_sentences(self.phrase1_tokenized,
                                              self.phrase2_tokenized, max_n=2)
        if self.cosine_similarity is None:
            self.cosine_similarity = \
                cosine_similarity_from_raw_sentences(self.phrase1,
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
                                        tokenizer: PreTrainedTokenizerBase):
    catchphrases = ds.data[0][SampleType.CATCHPHRASES]
    total = 0
    phrase_pair_diff_count: Dict[int, int] = defaultdict(lambda: 0)
    phrase_pair_differences: Dict[
        int, List[PhrasePairDifference]] = defaultdict(
        lambda: [])
    for idx, d in enumerate(ds.data):
        other_catchphrases = d[SampleType.CATCHPHRASES]
        for phrase in catchphrases:
            tokenized_phrase = tokenizer.tokenize(phrase)
            closest_diff = float('inf')
            phrase_pair_closest: Tuple[str, str] = None
            for other_catchphrase in other_catchphrases:
                tokenized_other_catchphrase = tokenizer.tokenize(
                    other_catchphrase)
                word_diff = word_count_diff(tokenized_phrase,
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
    from transformers import BertTokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    ds = AusCaseReports(tokenizer=tokenizer, max_tgt_len=1000,
                        max_src_len=10000).load_training_data()
    investigate_catchphrase_differences(ds, tokenizer)


if __name__ == '__main__':
    main()
