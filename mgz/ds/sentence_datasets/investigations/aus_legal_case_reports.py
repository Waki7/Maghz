from mgz.ds.sentence_datasets.aus_legal_case_reports import *


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


def inspect_catchphrase_diffs():
    # please install HuggingFace ds by pip install ds
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")

    ds = AusCaseReportsToTagGrouped(tokenizer=tokenizer,
                                    training_ratio=1.0).load_training_data()
    investigate_catchphrase_differences(ds, tokenizer)


def inspect_src_test():
    from transformers import BartTokenizer
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    ds = AusCaseReports(tokenizer=tokenizer, max_src_len=17000, max_tgt_len=256,
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
