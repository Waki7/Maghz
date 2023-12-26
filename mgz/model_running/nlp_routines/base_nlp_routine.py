from __future__ import annotations
from __future__ import annotations

import time

import torch.utils.data
import torch.utils.data
from tqdm import tqdm

import mgz.settings as settings
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class BaseProtocol(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def __init__(self):
        pass

    def run_epoch(self,
                  model_node: ModelNode,
                  data_loader: torch.utils.data.DataLoader[TagQAMetaTaskBatch],
                  val_data_loader: torch.utils.data.DataLoader[
                      TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge,
                  log_interval=5,
                  val_interval=50,
                  gradient_accumulation_steps=8,
                  debug=False,
                  ) -> ModelNode:
        """Train a single epoch"""
        model, tokenizer = model_node.model, model_node.tokenizer
        model.train()

        start = time.time()
        batch: Sent2TagMetaTaskBatch
        if not model_node.has_initial_metrics():
            model_node.store_metrics(
                self.val_model(val_data_loader, model_node, model_edge))
        optimizer = model_edge.optimizer
        scheduler = model_edge.scheduler
        batch_num = 0
        i: int

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if batch is None:
                logging.warning(
                    'batch is None... need to fix issue of tags are super sparse')
                continue
            batch_num += 1

            loss_w_grad: FloatTensorT['1']
            accuracy: float
            loss_w_grad, accuracy = self.run_batch(model, batch, model_edge)
            (loss_w_grad / gradient_accumulation_steps).backward()
            if (batch_num + 1) % gradient_accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
                if scheduler is not None: scheduler.step()
                model_edge.train_state.accum_step += 1

            model_edge.train_state.step += 1

            if (batch_num + 1) % log_interval == 0:
                lr = model_edge.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                       + "| Tasks / Sec: %7.1f | Learning Rate: %6.1e")
                      % (
                          i, i, loss_w_grad,
                          model_edge.train_state.step / elapsed,
                          lr))

            if (batch_num + 1) % val_interval == 0:
                self.val_model(val_data_loader, model_node, model_edge)
            settings.empty_cache()
            batch_num += 1
        return model_edge.complete_model_transition()

    @torch.no_grad()
    def val_model(self,
                  val_data_loader: torch.utils.data.DataLoader[
                      TagQAMetaTaskBatch],
                  model_node: ModelNode,
                  training_edge: ModelTransitionEdge = None
                  ) -> Dict[Metrics, Union[float, List[float]]]:
        model = model_node.model
        started_training = model.training
        model.eval()

        i: int
        batch: Sent2TagMetaTaskBatch

        # Number of times to sample per tag, there may be
        # different combinations of support and queries, so a higher number willg
        # give you a better idea of the accuracy per tag.
        accuracies: List[float] = []

        batch: Sent2TagMetaTaskBatch
        for i, batch in tqdm(enumerate(val_data_loader)):
            if batch is None:
                logging.warning(
                    'batch is None... need to fix issue of tags are super sparse')
                continue

            loss_w_grad: FloatTensorT['1']
            accuracy: float
            loss_w_grad, accuracy = self.run_batch(model, batch, training_edge)

            accuracies.append(accuracy.item())
            print(f'Val Accuracy Moving Avg: {np.mean(accuracies)}')
        if training_edge is not None:
            training_edge.record_validation(accuracies)
        if started_training:
            model.train()
        settings.empty_cache()
        return {
            Metrics.VAL_ACC_ALL: accuracies,
            Metrics.VAL_ACC_MEAN: np.mean(accuracies)
        }
