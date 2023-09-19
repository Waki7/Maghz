from __future__ import annotations

import time

import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import mgz.version_control as vc
import settings
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.run_ops import run_epoch, TrainState
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *
from mgz.version_control import ModelNode

writer = SummaryWriter()


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
        log_interval=1,
        accuracy_interval=1,
) -> (torch.Tensor, TrainState):
    """Train a single epoch"""

    start = time.time()
    total_queries = 0
    tokens = 0
    n_tags_trained_on = 0
    loss = 0
    iter_per_tags = 5
    n_query = 4
    i: int
    batch: Sent2TagMetaTaskBatch
    for i, batch in tqdm(enumerate(data_loader)):
        if batch is None:
            logging.warning(
                'batch is None... need to fix issue of tags are super sparse')
            continue
        tgt_tags = batch.tgt_tag_ids

        # treat each tag as a separate task
        for tag_idx in range(len(tgt_tags)):
            all_embeddings: FloatTensorT[
                'TaskSize,EmbedLen'] = model.forward(
                src_ids=batch.per_tag_src_ids[tag_idx],
                tgt_ids=batch.tgt_tag_ids[tag_idx],
                src_mask=batch.per_tag_masks[tag_idx],
                tgt_mask=batch.tgt_tag_masks[tag_idx]
            )

            loss = torch.tensor(0.0).to(all_embeddings.device)
            for task_iter in range(iter_per_tags):
                support: FloatTensorT['NClasses,EmbedLen']
                queries, support, query_lbls = batch.sample_task_from_batch(
                    all_embeddings, tag_idx,
                    n_query)

                distance_to_supports_per_cls: List[FloatTensorT['NQuery']] = []
                # TODO can probably do this in a batched way
                for cls in range(support.shape[0]):
                    distance_to_supports_per_cls.append(
                        torch.linalg.norm(
                            queries - support[cls, :], dim=-1, ord=2))
                distance_to_supports_per_cls: FloatTensorT['NQuery,NClasses'] = \
                    FloatTensorT(torch.stack(
                        distance_to_supports_per_cls, dim=-1))
                distance_to_supports_per_cls_probs = torch.softmax(
                    -1 * distance_to_supports_per_cls, dim=-1)
                loss += torch.nn.CrossEntropyLoss()(
                    distance_to_supports_per_cls_probs,
                    query_lbls) / (support.shape[0] + queries.shape[0])
                total_queries += queries.shape[0]
                train_state.step += 1

            loss.backward()
            writer.add_scalar('Loss/train', loss, train_state.step)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_tags_trained_on += 1
            train_state.accum_step += 1

        if scheduler is not None: scheduler.step()
        print('i', i)
        print('log_interval', log_interval)
        print('mod', i % log_interval)
        if (i + 1) % log_interval == 0:
            lr = optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                   + "| Tasks / Sec: %7.1f | Learning Rate: %6.1e")
                  % (
                      i, n_tags_trained_on, loss, train_state.step / elapsed,
                      lr))

        if (i + 1) % accuracy_interval == 0:
            settings.empty_cache()
            val_model(val_data_loader, model, tokenizer)

        # cuda / mps / gpu usage and managing
        settings.empty_cache()
        settings.print_gpu_usage()

    return train_state


@torch.no_grad()
def val_model(
        val_data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
        model: BaseTransformer,
        tokenizer: PreTrainedTokenizer
) -> (torch.Tensor, TrainState):
    started_training = model.training
    model.eval()

    if started_training:
        model.train()
    i: int
    batch: Sent2TagMetaTaskBatch
    n_query = 4

    # Number of times to sample per tag, there may be
    # different combinations of support and queries, so a higher number willg
    # give you a better idea of the accuracy per tag.
    iter_per_tags = 5

    for i, batch in tqdm(enumerate(val_data_loader)):
        if batch is None:
            logging.warning(
                'batch is None... need to fix issue of tags are super sparse')
            continue
        tgt_tags = batch.tgt_tag_ids

        # treat each tag as a separate task
        for tag_idx in range(len(tgt_tags)):
            all_embeddings: FloatTensorT[
                'TaskSize,EmbedLen'] = model.forward(
                src_ids=batch.per_tag_src_ids[tag_idx],
                tgt_ids=batch.tgt_tag_ids[tag_idx],
                src_mask=batch.per_tag_masks[tag_idx],
                tgt_mask=batch.tgt_tag_masks[tag_idx]
            )

            for tag_idx in range(iter_per_tags):
                support: FloatTensorT['NClasses,EmbedLen']
                queries, support, query_lbls = batch.sample_task_from_batch(
                    all_embeddings, tag_idx,
                    n_query)
                distance_to_supports_per_cls: List[FloatTensorT['NQuery']] = []
                # TODO can probably do this in a batched way
                for cls in range(support.shape[0]):
                    distance_to_supports_per_cls.append(
                        torch.linalg.norm(
                            queries - support[cls, :], dim=-1, ord=2))
                distance_to_supports_per_cls: FloatTensorT['NQuery,NClasses'] = \
                    FloatTensorT(torch.stack(
                        distance_to_supports_per_cls, dim=-1))
                predictions: LongTensorT['NQuery'] = torch.softmax(
                    -1 * distance_to_supports_per_cls, dim=-1).argmax(-1)
                accuracy = (predictions == query_lbls).float().mean()
                writer.add_scalar('Accuracy/val', accuracy)


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
