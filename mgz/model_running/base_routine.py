from __future__ import annotations

import json
import time

import torch.utils.data
from tqdm import tqdm
from transformers import PreTrainedTokenizer

import mgz.metrics.nlp.metrics as metrics
import mgz.settings as settings
import mgz.version_control as vc
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2SentBatch, \
    Sent2TagMetaTaskBatch, TagQAMetaTaskBatch
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *

if TYPE_CHECKING:
    from mgz.version_control import ModelTransitionEdge, ModelNode, Metrics


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
        log_interval=40,
        accuracy_interval=120,
        model_edge: ModelTransitionEdge = None,
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
        model_edge.train_state.step += 1
        model_edge.train_state.samples += batch.src.shape[0]
        model_edge.train_state.tokens += batch.ntokens
        if i % accum_iter == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            n_accum += 1
            model_edge.train_state.accum_step += 1
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
    return total_loss / total_tokens, model_edge.train_state


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

    @abstractmethod
    def _check(self, ds: BaseDataset):
        raise NotImplementedError

    @abstractmethod
    def run_batch(self, model: BaseTransformer,
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge,
                  gpu_max_batch_size=4) -> \
            Tuple[FloatTensorT['1'], float]:
        raise NotImplementedError

    @abstractmethod
    def train_epoch(self,
                    model_node: ModelNode,
                    data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    val_data_loader: torch.utils.data.DataLoader[
                        TagQAMetaTaskBatch],
                    model_edge: ModelTransitionEdge,
                    log_interval=5,
                    val_interval=50,
                    gradient_accumulation_steps=8,
                    debug=False,
                    ) -> ModelNode:
        raise NotImplementedError

    @abstractmethod
    @torch.no_grad()
    def val_model(self,
                  val_data_loader: torch.utils.data.DataLoader[
                      TagQAMetaTaskBatch],
                  model_node: ModelNode,
                  training_edge: ModelTransitionEdge = None
                  ) -> Dict[Metrics, Union[float, List[float]]]:
        raise NotImplementedError

    @abstractmethod
    def predict(self, model_node: vc.ModelNode):
        raise NotImplementedError

    def train(self, model_node: vc.ModelNode, ds: BaseDataset,
              model_edge: vc.ModelTransitionEdge, device=None,
              distributed: bool = False,
              turn_off_shuffle=False,
              val_ds: BaseDataset = None, n_epochs=1) -> vc.ModelNode:
        model_node.model.train()
        self._check(ds)

        if val_ds is None:
            val_ds = ds.gen_validation_data()
        else:
            val_ds = val_ds.load_validation_data()
        train_ds = ds.load_training_data()

        if device is None:
            device = settings.DEVICE

        train_dl = train_ds.create_dataloaders(device, batch_size=1,
                                               is_distributed=distributed,
                                               turn_off_shuffle=turn_off_shuffle)
        val_dl = val_ds.create_dataloaders(device, batch_size=1,
                                           is_distributed=distributed,
                                           turn_off_shuffle=turn_off_shuffle)
        for i in range(0, n_epochs):
            self.train_epoch(
                model_node,
                data_loader=train_dl,
                val_data_loader=val_dl,
                model_edge=model_edge
            )
        return model_node

    def evaluate(self, model_node: ModelNode, val_ds: BaseDataset,
                 device=None, )->Dict[Metrics, Union[float, List[float]]]:
        if device is None:
            device = settings.DEVICE
        val_ds.load_validation_data()
        val_dl = val_ds.create_dataloaders(device, batch_size=1,
                                           is_distributed=False,
                                           turn_off_shuffle=True)
        metrics: Dict[Metrics, Union[float, List[float]]] = self.val_model(val_dl, model_node)
        return metrics

