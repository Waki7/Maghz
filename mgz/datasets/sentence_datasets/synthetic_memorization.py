import spaces as sp
from mgz.datasets.sentence_datasets.sentence_datasets import SentenceDataset, \
    SentenceBatch
from mgz.typing import *


class SyntheticMemorization(SentenceDataset):
    def __init__(self, vocab_size: int, out_vocab_size: int, max_length: int,
                 n_samples: int, batch_size: int):
        super(SyntheticMemorization, self).__init__()
        self.input_space = sp.Sentence(vocab_size, shape=(max_length,))
        self.target_space = sp.Sentence(out_vocab_size, shape=(max_length,))

        self.max_length = max_length
        self.n_batches = n_samples // batch_size
        self.batch_size = batch_size

        self.use_cuda = False

    def cuda(self):
        self.use_cuda = True
        return self

    def cpu(self):
        self.use_cuda = False
        return self

    def __len__(self):
        'Denotes the total number of samples'
        return len(self.img_index)

    def gen(self) -> Generator[SentenceBatch, None, None]:
        for i in range(self.batch_size):
            data = torch.randint(1, self.input_space.vocab_size,
                                 size=(self.batch_size, self.max_length))
            data[:, 0] = 1
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
