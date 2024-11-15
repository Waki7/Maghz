from __future__ import annotations

import torch.utils.data
import transformers as hug
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm import tqdm

import mgz.settings as settings
from mgz.ds.sentence_datasets.datasets_base.sentence_batch import SentenceBatch
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
                 gradient_accumulation_steps: int = 4,
                 debug: bool = False,
                 ):
        self.tokenizer = tokenizer
        self.debug = debug
        self.gpu_max_batch_size = gpu_max_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps

        # To get configured
        self.configured = False
        self.lr = None
        self.weight_decay = None
        self.betas = None
        self.eps = None
        self.loss_fn = None
        self.scheduler = None
        self.optimizer = None

    def configure(self, optimizer: torch.optim.Optimizer, lr: float,
                  weight_decay: float,
                  betas: Tuple[float, float] = (0.9, 0.999),
                  eps: float = 1e-8,
                  scheduler: torch.optim.lr_scheduler.LRScheduler = None,
                  loss_fn: Callable = None):
        self.lr = lr
        self.weight_decay = weight_decay
        self.betas = betas
        self.eps = eps
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.scheduler = scheduler
        self.configured = True

    def debug_log_batch_info(self, batch: SentenceBatch):
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
                        SentenceBatch],
                    val_data_loader: torch.utils.data.DataLoader[
                        SentenceBatch],
                    model_edge: ModelTransitionEdge,
                    log_interval=5,
                    val_interval=50,
                    debug=False,
                    ) -> ModelNode:
        """Train a single epoch"""
        model, tokenizer = model_node.model, model_node.tokenizer
        model.train()
        set_seed(0)

        if not model_node.has_initial_metrics():
            model_node.store_metrics(
                self.val_model(val_data_loader, model_node, model_edge))
        optimizer = model_edge.optimizer
        scheduler = model_edge.scheduler
        batch_num: int = 0
        best_val_acc: float = 0.0
        model_edge.start_timer()

        accelerator = Accelerator(
            gradient_accumulation_steps=self.gradient_accumulation_steps)
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
                      SentenceBatch],
                  model_node: ModelNode,
                  model_edge: ModelTransitionEdge = None
                  ) -> Dict[Metrics, Union[float, List[float]]]:
        model = model_node.model
        started_training = model.training
        model.eval()

        # Number of times to sample per tag, there may be
        # different combinations of support and queries, so a higher number willg
        # give you a better idea of the accuracy per tag.
        accuracies: List[float] = []
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
