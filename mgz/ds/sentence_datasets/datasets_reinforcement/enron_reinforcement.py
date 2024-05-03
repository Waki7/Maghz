from functools import partial

from transformers import LlamaTokenizer, PreTrainedTokenizerBase

import spaces as sp
from mgz.ds.sentence_datasets.datasets_base.enron_emails import EnronEmails
from mgz.ds.sentence_datasets.datasets_base.sentence_datasets import \
    SentenceDataset, \
    SampleType
from mgz.ds.sentence_datasets.datasets_reinforcement.reinforcement_batch import \
    ReinforcementBatch
from mgz.ds.sentence_datasets.gpt_input_augments import PromptConfig, \
    BatchChatInput, DocumentRequestChat
from mgz.typing import *


class EnronReinforcement(EnronEmails):

    def __init__(self, tokenizer: Union[
        LlamaTokenizer, LlamaTokenizer, PreTrainedTokenizerBase],
                 max_src_len: SrcSeqLen,
                 prompt_config: PromptConfig,
                 training_ratio: float = 0.75,
                 dataset_dir: str = None):
        assert isinstance(tokenizer, LlamaTokenizer) or isinstance(
            tokenizer, LlamaTokenizer), \
            f'This dataset requires the LLamaTokenizer found {type(tokenizer)}'
        super(EnronReinforcement, self).__init__(tokenizer=tokenizer,
                                                 max_src_len=max_src_len,
                                                 dataset_dir=dataset_dir,
                                                 training_ratio=training_ratio)
        self.tags: Set
        self._prompt_config = prompt_config

    def _load(self, train: bool = False, val: bool = False, test: bool = False):
        super()._load(train, val, test)  # Only loads into self.data
        self.loaded = False
        keys_to_keep = [
            SampleType.CATCHPHRASES,
            SampleType.FULL_AS_STRING,
        ]
        catchphrases = set()
        for i in range(len(self.data)):
            doc_filtered = {key: self.data[i][key] for key in keys_to_keep}
            self.data[i] = doc_filtered
            catchphrases.update(self.data[i][SampleType.CATCHPHRASES])
        self.tags = catchphrases
        self.loaded = True

    @property
    @overrides(SentenceDataset)
    def input_space(self) -> sp.Sentence:
        return sp.Sentence(
            len((self.vocab_src)), sequence_len=self.max_src_len)

    def __len__(self):
        'Denotes the total number of samples'
        assert self.loaded, "Dataset not loaded"
        return len(self.data)

    @property
    @overrides(SentenceDataset)
    def target_space(self) -> sp.Tagging:
        return sp.BinaryTagging()

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(ReinforcementBatch.default_collate_fn, self, device)

    @overrides(EnronEmails)
    def __getitem__(self, idx) -> BatchChatInput:
        tags_as_list = list(self.tags)[0]
        prompt_input: BatchChatInput = \
            DocumentRequestChat(
                prompt_config=self.prompt_config,
                document_text=self.data[idx][SampleType.FULL_AS_STRING],
                document_requests=tags_as_list)
        return prompt_input
