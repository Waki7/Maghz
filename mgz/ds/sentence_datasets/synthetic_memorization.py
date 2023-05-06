from functools import partial

import spaces as sp
from mgz.ds.sentence_datasets.sentence_datasets import SentenceDataset, \
    SentenceBatch
from mgz.models.nlp.tokenizing import tokenize
from mgz.typing import *


class SyntheticMemorization(SentenceDataset):
    def __init__(self, vocab_size: int, out_vocab_size: int, max_length: int,
                 n_samples: int, batch_size: int):
        super(SyntheticMemorization, self).__init__()
        self.input_space = sp.Sentence(vocab_size, shape=(max_length,))
        self.target_space = sp.Sentence(out_vocab_size, shape=(max_length,))

        self.max_length = max_length
        self.n_samples = n_samples
        self.n_batches = n_samples // batch_size
        self.batch_size = batch_size
        self.loaded = True
        self.use_cuda = False

    def cuda(self):
        self.use_cuda = True
        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def __len__(self):
        'Denotes the total number of samples'
        return self.n_samples

    def gen(self) -> Generator[SentenceBatch, None, None]:
        for i in range(self.batch_size):
            data = torch.randint(1, self.input_space.vocab_size,
                                 size=(self.batch_size, self.max_length))
            data[:, 0] = 1  # starting token
            src = data.requires_grad_(False).clone().detach()
            tgt = data.requires_grad_(False).clone().detach()
            batch = SentenceBatch(src, tgt, 0)
            if self.use_cuda:
                yield batch.cuda()
            yield batch

    def __getitem__(self, index) -> SentenceBatch:
        '''
        Generates one sample of data

        :param index:
        :return: an image and a regression target, the regression target is 5 long, as per the __init__ def of the space
        '''
        batch: SentenceBatch = next(self.gen())
        return batch

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[SentenceBatch]):
        assert self.loaded, "Dataset not loaded"
        func = torch.cat if len(batch[0].src.shape) == 2 else torch.stack
        src = func([b.src for b in batch])
        tgt = func([b.src for b in batch])
        return src, tgt

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)

    def tokenize_src(self, text) -> List[str]:
        return tokenize(text, self.tokenizer_src)

    def tokenize_tgt(self, text) -> List[str]:
        return tokenize(text, self.tokenizer_tgt)

    def pad_idx(self) -> int:
        return -1

    def src_vocab_len(self) -> int:
        return self.input_space.vocab_size

    def tgt_vocab_len(self) -> int:
        return self.target_space.vocab_size
