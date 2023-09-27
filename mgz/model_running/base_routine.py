from __future__ import annotations

import json
import time

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import mgz.models.nlp.metrics as metrics
import mgz.version_control as vc
import settings
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2SentBatch
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

class TrainState:
    """Track number of steps, examples, and tokens processed"""

    step: int = 0  # Steps in the current epoch
    accum_step: int = 0  # Number of gradient accumulation steps
    samples: int = 0  # total # of examples used
    tokens: int = 0  # total # of tokens processed

    def to_json(self, as_str=False) -> Union[dict, str]:
        obj_dict = {}
        for k, v in sorted(self.__dict__.items()):
            obj_dict[k] = v
        return json.dumps(obj_dict, indent=4,
                          separators=(',', ': ')) if as_str else obj_dict


def run_epoch(
        data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        val_data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        model: BaseTransformer,
        tokenizer: PreTrainedTokenizer,
        loss_fn,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.LRScheduler = None,
        accum_iter=1,
        train_state=TrainState(),
        log_interval=40,
        accuracy_interval=120,
) -> (torch.Tensor, TrainState):
    """Train a single epoch"""
    start = time.time()
    total_tokens = 0
    total_loss = 0
    tokens = 0
    n_accum = 0
    i: int
    batch: Sent2SentBatch
    with torch.no_grad():
        val_model(val_data_loader, model, tokenizer)

    for i, batch in tqdm(enumerate(data_loader)):
        logits = model.forward(
            src_ids=batch.src, tgt_ids=batch.tgt, src_mask=batch.src_mask,
            tgt_mask=batch.tgt_mask
        )

        loss, loss_node = loss_fn(logits, batch.tgt, batch.ntokens)
        # loss_node = loss_node / accum_iter
        loss_node.backward()
        train_state.step += 1
        train_state.samples += batch.src.shape[0]
        train_state.tokens += batch.ntokens
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            train_state.accum_step += 1
        if scheduler is not None: scheduler.step()

        total_loss += loss
        total_tokens += batch.ntokens
        tokens += batch.ntokens
        if i % log_interval == 1:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(
                (
                        "Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                        + "| Tokens / Sec: %7.1f | Learning Rate: %6.1e"
                )
                % (i, n_accum, loss / batch.ntokens, tokens / elapsed, lr)
            )
            start = time.time()
            tokens = 0
        if i % accuracy_interval == 1:
            settings.empty_cache()
            val_model(val_data_loader, model, tokenizer)
        # cuda / mps / gpu usage and managing
        settings.empty_cache()
        settings.print_gpu_usage()
    return total_loss / total_tokens, train_state


@torch.no_grad()
def val_model(
        val_data_loader: torch.utils.data.DataLoader[Sent2SentBatch],
        model: BaseTransformer,
        tokenizer: PreTrainedTokenizer
) -> (torch.Tensor, TrainState):
    started_training = model.training
    model.eval()

    if started_training:
        model.train()
    i: int
    batch: Sent2SentBatch
    n_samples = 0
    # TODO, MAKE SURE THAT BLEU SCORES ADJACENTLY, CHECK BY EITHER VARYING THE BATCH SIZE AND MAKIGN SURE SAME SCORE, OR SHUFFLYING
    for i, batch in tqdm(enumerate(val_data_loader)):
        # TgtSeqLen is max padded length
        predictions: LongTensorT['B,TgtSeqLen'] = model.generate(
            src_ids=batch.src, src_mask=batch.src_mask)
        prediction_sentences: List[str] = tokenizer.batch_decode(predictions,
                                                                 skip_special_tokens=True)
        tgt_sentences: List[str] = tokenizer.batch_decode(batch.tgt,
                                                          skip_special_tokens=True)
        print(("Validation BLEU Score: %6d")
              % (metrics.bleu_from_tokenized_sentences(prediction_sentences,
                                                       tgt_sentences)))


class BaseProtocol(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelTransitionEdge,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> vc.ModelNode:
        pass

    def evaluate(self):
        pass

    def predict(self):
        pass
