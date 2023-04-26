from __future__ import annotations

import multiprocessing as mp
import os
import time

import GPUtil
import altair as alt
import pandas as pd
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum
from mgz.ds.sentence_datasets.sentence_datasets import SentenceBatch, \
    SentenceDataset
from mgz.models.nlp.bert_basic import subsequent_mask, EncoderDecoder, \
    PredictorHead, make_model
from mgz.typing import *
def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (
            model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))
    )


def example_learning_schedule():
    opts = [
        [512, 1, 4000],  # example 1
        [512, 1, 8000],  # example 2
        [256, 1, 4000],  # example 3
    ]

    dummy_model = torch.nn.Linear(1, 1)
    learning_rates = []

    # we have 3 examples in opts list.
    for idx, example in enumerate(opts):
        # run 20000 epoch for each example
        optimizer = torch.optim.Adam(
            dummy_model.parameters(), lr=1, betas=(0.9, 0.98), eps=1e-9
        )
        lr_scheduler = LambdaLR(
            optimizer=optimizer, lr_lambda=lambda step: rate(step, *example)
        )
        tmp = []
        # take 20K dummy training steps, save the learning rate at each step
        for step in range(20000):
            tmp.append(optimizer.param_groups[0]["lr"])
            optimizer.step()
            lr_scheduler.step()
        learning_rates.append(tmp)

    learning_rates = torch.tensor(learning_rates)

    # Enable altair to handle more than 5000 rows
    alt.data_transformers.disable_max_rows()

    opts_data = pd.concat(
        [
            pd.DataFrame(
                {
                    "Learning Rate": learning_rates[warmup_idx, :],
                    "model_size:warmup": ["512:4000", "512:8000", "256:4000"][
                        warmup_idx
                    ],
                    "step": range(20000),
                }
            )
            for warmup_idx in [0, 1, 2]
        ]
    )

    return (
        alt.Chart(opts_data)
        .mark_line()
        .properties(width=600)
        .encode(x="step", y="Learning Rate", color="model_size:warmup:N")
        .interactive()
    )


example_learning_schedule()


class LabelSmoothing(nn.Module):
    '''
    Implement label smoothing. This is kind of like saying instead of strongly
    tell the matter class a is completely right (set to 1) and everything else
    is completely wrong (set to 0), this gives a more small directonal nudge.
    '''

    def __init__(self, n_cls: int, padding_idx: int, smoothing=0.0):
        super(LabelSmoothing, self).__init__()
        self.criterion = nn.KLDivLoss(reduction="sum")
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.n_cls = n_cls
        self.true_dist = None

    def forward(self, x: FloatTensorT['B*SrcSeqLen,OutNClasses'],
                target: FloatTensorT['B*SrcSeqLen']):
        assert x.size(1) == self.n_cls
        true_dist: FloatTensorT['B*SrcSeqLen,OutNClasses'] = torch.ones_like(x)
        true_dist *= (self.smoothing / (self.n_cls - 2))
        true_dist.scatter_(1, target.data.unsqueeze(1).long(), self.confidence)
        true_dist[:, self.padding_idx] = 0
        mask = torch.nonzero(target.data == self.padding_idx)
        if mask.dim() > 0:
            true_dist.index_fill_(0, mask.squeeze(), 0.0)
        self.true_dist = true_dist
        return self.criterion(x, true_dist.clone().detach())


class SimpleLossCompute:
    "A simple loss compute and train function."

    def __init__(self, generator: PredictorHead, criterion):
        self.generator = generator
        self.criterion = criterion

    def __call__(self, x: FloatTensorT['B,SrcSeqLen,EmbedLen'],
                 y: FloatTensorT['B,SrcSeqLen'],
                 norm_by: int):
        x: FloatTensorT['B,SrcSeqLen,OutNClasses'] = self.generator(x)
        sloss = (
                self.criterion(
                    x.contiguous().view(-1, x.size(-1)), y.contiguous().view(-1)
                )
                / norm_by
        )
        return sloss.data * norm_by, sloss


class DummyOptimizer(torch.optim.Optimizer):
    def __init__(self):
        self.param_groups = [{"lr": 0}]
        None

    def step(self):
        None

    def zero_grad(self, set_to_none=False):
        None


class DummyScheduler:
    def step(self):
        None
