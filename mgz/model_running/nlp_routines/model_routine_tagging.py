from __future__ import annotations

import time

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import mgz.models.nlp.metrics as metrics
import mgz.version_control as vc
import settings
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.version_control import ModelNode
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.run_ops import run_epoch, TrainState
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *


def run_epoch(
        data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
        val_data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
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
    batch: Sent2TagMetaTaskBatch
    for i, batch in tqdm(enumerate(data_loader)):
        pos_srcs = batch.per_tag_pos_srcs
        neg_srcs = batch.per_tag_neg_srcs
        tgt_tags = batch.tgt_tag

        # treat each tag as a separate task
        for task_idx, tag in enumerate(tgt_tags):
            pos_src = pos_srcs[task_idx]
            logits = model.forward(
                src_ids=pos_src, tgt_ids=batch.tgt, src_mask=batch.src_mask,
                tgt_mask=batch.tgt_mask
            )

            neg_src = neg_srcs[task_idx]
            logits = model.forward(
                src_ids=pos_src, tgt_ids=batch.tgt, src_mask=batch.src_mask,
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


class TaggingRoutine(BaseProtocol):
    def __init__(self):
        super().__init__()
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelEdge,
              batch_size=8, device=None, distributed: bool = False,
              turn_off_shuffle=False) -> vc.ModelNode:
        # from mgz.ds.sentence_datasets.aus_legal_case_reports import TagGroupedSampler
        model_node.model.train()
        if model_node.metrics is None:
            pass
        self._check(ds)

        val_ds = ds.gen_validation_data()
        train_ds = ds.load_training_data()

        if device is None:
            device = settings.DEVICE

        meta_source_batch_size = 1
        train_dl = train_ds.create_dataloaders(device, meta_source_batch_size,
                                               is_distributed=distributed,
                                               turn_off_shuffle=turn_off_shuffle)
        val_dl = val_ds.create_dataloaders(device, meta_source_batch_size,
                                           is_distributed=distributed,
                                           turn_off_shuffle=turn_off_shuffle)
        run_epoch(
            data_loader=train_dl,
            val_data_loader=val_dl,
            model=model_node.model, tokenizer=model_node.tokenizer,
            loss_fn=model_edge.loss_fn,
            optimizer=model_edge.optimizer,
            train_state=model_edge.train_state,
        )
        return model_node

    def evaluate(self, model_node: ModelNode):
        pass

    def predict(self, model_node: ModelNode):
        pass
