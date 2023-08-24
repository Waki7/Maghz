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
import settings
import torch
from transformers import BartTokenizer

import settings
from mgz.models.nlp.bart import BartForConditionalGeneration
from mgz.model_running.run_ops import generate_controller, forward_controller
from mgz.typing import *
import mgz.models.nlp.bart_orig as hug
from transformers import GenerationConfig, BartConfig

from mgz.ds.sentence_datasets.multi_lex_sum import MultiLexSum
from mgz.ds.sentence_datasets.sentence_datasets import Sent2SentBatch, \
    SentenceDataset
from mgz.models.nlp.bert_basic import subsequent_mask, EncoderDecoder, \
    PredictorHead, make_model
from mgz.typing import *
from mgz.models.nlp.bart import BartForConditionalGeneration

from mgz.model_vc.model_index import Indexer


def main():
    model_name = 'allenai/bart-large-multi_lexsum-long-short'
    model_mgz: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
        model_name).to(settings.DEVICE)

    idxer2 = Indexer.load_from_json()
    print(idxer2.to_json())


if __name__ == '__main__':
    main()
