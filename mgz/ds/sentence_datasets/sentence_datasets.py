from functools import partial

from torch.nn.functional import pad
from torchtext.vocab.vocab import Vocab

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.typing import *

class SentenceBatch:
    """Object for holding a batch of data with mask during training."""

    def __init__(self, src: LongTensorT['B,SrcSeqLen'],
                 tgt: SentenceT['B,SeqLen - 1'] = None,
                 pad=2):  # 2 = <blank>
        self.src: LongTensorT['B,SrcSeqLen'] = src
        self.src_mask = (src != pad).unsqueeze(-2)
        if tgt is not None:
            self.tgt: LongTensorT['B,SeqLen - 1'] = tgt
            self.tgt_y = tgt[:, 1:]
            self.tgt_mask = self.make_std_mask(self.tgt, pad)
            self.ntokens = (self.tgt_y != pad).data.sum()
        self.tgt_mask.to(self.tgt.device)
        self.src_mask.to(self.src.device)

    @staticmethod
    def make_std_mask(tgt: LongTensorT['B,SrcSeqLen'], pad: int):
        "Create a mask to hide padding and future words."
        tgt_mask = (tgt != pad).unsqueeze(-2)
        tgt_mask = tgt_mask & subsequent_mask(tgt.size(-1)).type_as(
            tgt_mask.data
        )
        return tgt_mask

    def try_cuda(self):
        self.src = self.src.cuda()
        self.src_mask = self.src_mask.cuda()
        self.tgt = self.tgt.cuda()
        self.tgt_y = self.tgt_y.cuda()
        self.tgt_mask = self.tgt_mask.cuda()
        return self


class SentenceDataset(BaseDataset):
    def __init__(self):
        super(SentenceDataset, self).__init__()
        self.input_space = None
        self.target_space = None

        # --- Initialization flags ---
        self.use_cuda = False
        self.loaded = False

    @property
    def in_space(self) -> sp.Sentence:
        raise NotImplementedError

    @property
    def pred_space(self) -> Union[sp.Sentence, sp.RegressionTarget]:
        raise NotImplementedError

    def _collate_fn(self, device: Union[int, torch.device],
                    batch: List[Tuple[GermanT, EnglishT]]):
        raise NotImplementedError

    def get_collate_fn(self, device: Union[int, torch.device]):
        assert self.loaded, "Dataset not loaded"
        return partial(self._collate_fn, device)


def subsequent_mask(size: SrcSeqLen):
    "Mask out subsequent positions."
    attn_shape = (1, size, size)
    subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(
        torch.uint8
    )
    return subsequent_mask == 0


def collate_batch(
        batch,
        src_pipeline: Callable[[SrcStringT], List[str]],  # tokenizer function
        tgt_pipeline: Callable[[TgtStringT], List[str]],  # tokenizer function
        src_vocab: Vocab,
        tgt_vocab: Vocab,
        device,
        max_padding=128,
        pad_id=2,
) -> Tuple[LongTensorT['B,SrcSeqLen'], LongTensorT['B,OutSeqLen']]:
    dtype = torch.int32
    bs_id = torch.tensor([0], device=device, dtype=dtype)  # <s> token id
    eos_id = torch.tensor([1], device=device, dtype=dtype)  # </s> token id
    src_list, tgt_list = [], []
    for (_src, _tgt) in batch:
        processed_src = torch.cat(
            [
                bs_id,
                torch.tensor(
                    src_vocab.forward(src_pipeline(_src)),
                    dtype=dtype,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        processed_tgt = torch.cat(
            [
                bs_id,
                torch.tensor(
                    tgt_vocab.forward(tgt_pipeline(_tgt)),
                    dtype=dtype,
                    device=device,
                ),
                eos_id,
            ],
            0,
        )
        src_list.append(
            # warning - overwrites values for negative values of padding - len
            pad(
                processed_src,
                (
                    0,
                    max_padding - len(processed_src),
                ),
                value=pad_id,
            )
        )
        tgt_list.append(
            pad(
                processed_tgt,
                (0, max_padding - len(processed_tgt)),
                value=pad_id,
            )
        )

    src = torch.stack(src_list)
    tgt = torch.stack(tgt_list)
    return (src, tgt)
