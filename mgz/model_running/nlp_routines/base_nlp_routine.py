from __future__ import annotations

import torch.utils.data
import transformers as hug
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

import mgz.settings as settings
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class BaseNLPProtocol(BaseProtocol):
    __metaclass__ = ABCMeta

    def __init__(self,
                 tokenizer: hug.PreTrainedTokenizerBase = None,
                 gpu_max_batch_size: int = 4,
                 debug: bool = False,
                 ):
        self.tokenizer = tokenizer
        self.debug = debug
        self.gpu_max_batch_size = gpu_max_batch_size

    def debug_log_batch_info(self, batch: TagQAMetaTaskBatch):
        if self.debug:
            print(
                '_____________________________________________________________________')
            print(f' First email that has a {batch.query_lbls[0]} tag-----')
            print(self.tokenizer.batch_decode(batch.queries,
                                              skip_special_tokens=True)[0])
            print(f' Second email that has a {batch.query_lbls[1]} tag-----')
            print(self.tokenizer.batch_decode(batch.queries,
                                              skip_special_tokens=True)[1])

    def debug_log_predictions(self, similarity_to_classes: FloatTensorT[
        'NQuery,EmbedLen']):
        if self.debug:
            print('predictions ', torch.argmax(similarity_to_classes, dim=-1))

    @overrides(BaseProtocol)
    def train_epoch(self,
                    model_node: ModelNode,
                    data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    val_data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    model_edge: ModelTransitionEdge,
                    log_interval=5,
                    val_interval=50,
                    gradient_accumulation_steps=4,
                    debug=False,
                    ) -> ModelNode:
        """Train a single epoch"""
        model, tokenizer = model_node.model, model_node.tokenizer
        model.train()
        set_seed(0)

        batch: Sent2TagMetaTaskBatch
        if not model_node.has_initial_metrics():
            model_node.store_metrics(
                self.val_model(val_data_loader, model_node, model_edge))
        optimizer = model_edge.optimizer
        scheduler = model_edge.scheduler
        batch_num: int = 0
        best_val_acc: float = 0.0
        model_edge.start_timer()

        accelerator = Accelerator(
            gradient_accumulation_steps=gradient_accumulation_steps)
        model, optimizer, data_loader, scheduler = accelerator.prepare(
            model, optimizer, data_loader, scheduler
        )

        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if batch is None:
                logging.warning(
                    'batch is None... need to fix issue of tags are super sparse')
                continue
            batch_num += 1

            with accelerator.accumulate(model):

                loss_w_grad: FloatTensorT['1']
                accuracy: float
                loss_w_grad, accuracy = self.run_batch(model, batch, model_edge)

                # (loss_w_grad / gradient_accumulation_steps).backward()
                # if (batch_num) % gradient_accumulation_steps == 0 or (
                # batch_num) % val_interval == 0:
                accelerator.backward(loss_w_grad)
                optimizer.step()
                if scheduler is not None: scheduler.step()
                optimizer.zero_grad()
                model_edge.train_state.accum_step += 1

            model_edge.train_state.step += 1

            if (batch_num) % log_interval == 0:
                model_edge.print_train_step_info()

            if (batch_num) % val_interval == 0:
                acc_val_mean = \
                self.val_model(val_data_loader, model_node, model_edge)[
                    Metrics.VAL_ACC_MEAN]
                if acc_val_mean > best_val_acc:
                    model_edge.store_with_identifier("BEST_VAL",
                                                     {"val_acc": acc_val_mean})
                best_val_acc = max(best_val_acc, acc_val_mean)
            settings.empty_cache()
        return model_edge.complete_model_transition()

    @torch.no_grad()
    def val_model(self,
                  val_data_loader: torch.utils.data.DataLoader[
                      TagQAMetaTaskBatch],
                  model_node: ModelNode,
                  model_edge: ModelTransitionEdge = None
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
            loss_w_grad, accuracy = self.run_batch(model, batch, model_edge)

            accuracies.append(accuracy)
            print(f'Val Accuracy Moving Avg: {np.mean(accuracies)}')
        if model_edge is not None:
            model_edge.record_validation(accuracies)
        if started_training:
            model.train()
        settings.empty_cache()
        return {
            Metrics.VAL_ACC_ALL: accuracies,
            Metrics.VAL_ACC_MEAN: np.mean(accuracies)
        }
