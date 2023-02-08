import spaces as sp
from mgz.datasets.base_dataset import BaseDataset
from mgz.typing import *


class SentenceDataset(BaseDataset):
    def __init__(self):
        super(SentenceDataset, self).__init__()

    @property
    def in_space(self) -> sp.Sentence:
        raise NotImplementedError

    @property
    def pred_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        raise NotImplementedError


def subsequent_mask(size: SeqLen):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


class SentenceBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: SentenceT['B,SeqLen'],
                 tgt: SentenceT['B,SeqLen'] = None,
                 pad=2):  # 2 = <blank>
        self.src: SentenceT['B,SeqLen'] = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt: SentenceT['B,SeqLen'] = tgt[:, :-1]
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()

    @staticmethod
    def make_std_mask(tgt: SentenceT['B,SeqLen'], pad: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    def cuda(self):
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.tgt = self.tgt.cuda()
        self.tgt_y = self.tgt_y.cuda()
        self.tgt_mask = self.tgt_mask.cuda()
        return self
