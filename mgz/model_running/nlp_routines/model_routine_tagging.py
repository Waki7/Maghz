from __future__ import annotations

import time

import torch.utils.data
from tqdm import tqdm

import log_utils
import mgz.version_control as vc
import settings
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.run_ops import run_epoch, TrainState
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


def run_epoch(
        model_node: ModelNode,
        data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
        val_data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
        model_edge: ModelTransitionEdge,
        log_interval=5,
        val_interval=50,
) -> ModelNode:
    """Train a single epoch"""
    model, tokenizer = model_node.model, model_node.tokenizer
    model.train()

    start = time.time()
    loss = 0
    i: int
    batch: Sent2TagMetaTaskBatch
    loss_fn = torch.nn.CrossEntropyLoss()
    # loss_fn = LabelSmoothing(
    #     n_cls=2, padding_idx=model_node.tokenizer.pad_token_id,
    #     smoothing=0.1).to(settings.DEVICE)
    if not model_node.has_initial_metrics():
        val_model(val_data_loader, model)
        model_node.store_metrics({
            Metrics.VAL_ACC:
                log_utils.exp_tracker.get_mean_scalar(Metrics.VAL_ACC)},
            iter_metrics={
                Metrics.VAL_ACC: log_utils.exp_tracker.pop_scalars(
                    Metrics.VAL_ACC)
            })
    loss_interval = 5
    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch is None:
            logging.warning(
                'batch is None... need to fix issue of tags are super sparse')
            continue
        with torch.no_grad():
            n_cls, tsk_sz, _ = batch.supports.shape
            support_embeds: FloatTensorT[
                'TaskSize,EmbedLen'] = model.forward(
                src_ids=batch.supports.flatten(0, 1),
                tgt_ids=batch.tgt_tag_ids_supports.flatten(0, 1),
                src_mask=batch.support_masks.flatten(0, 1),
                tgt_mask=batch.tgt_tag_masks_supports.flatten(0, 1)
            )
            support_embeds: FloatTensorT[
                'NClasses,TaskSize,EmbedLen'] = \
                support_embeds.reshape(n_cls, tsk_sz, -1)
            support_centers: FloatTensorT['NClasses,EmbedLen'] = \
                support_embeds.mean(dim=1, keepdim=False)

        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.forward(
            src_ids=batch.queries,
            tgt_ids=batch.tgt_tag_ids_queries,
            src_mask=batch.query_masks,
            tgt_mask=batch.tgt_tag_masks_queries
        )
        query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
        loss = torch.tensor(0.0).to(query_embeds.device)
        distance_to_supports_per_cls: List[FloatTensorT['NQuery']] = []
        # TODO can probably do this in a batched way
        for cls in range(support_embeds.shape[0]):
            distance_to_supports_per_cls.append(
                FloatTensorT(torch.cosine_similarity(query_embeds,
                                                     support_centers[cls, :],
                                                     dim=-1))
                # -1 * torch.linalg.norm(
                #     query_embeds - support_centers[cls, :], dim=-1, ord=2)
            )
        distance_to_supports_per_cls: FloatTensorT['NQuery,NClasses'] = \
            FloatTensorT(torch.stack(
                distance_to_supports_per_cls, dim=-1))
        distance_to_supports_per_cls_probs = torch.softmax(
            distance_to_supports_per_cls, dim=-1)

        # Calculate accuracy
        predictions: LongTensorT[
            'NQuery'] = distance_to_supports_per_cls_probs.argmax(-1)
        log_utils.exp_tracker.add_scalar(Metrics.TRAIN_ACC_MEAN, (
                predictions == query_lbls).float().mean().item(),
                                         track_mean=True)
        loss += loss_fn.__call__(distance_to_supports_per_cls_probs,
                                 query_lbls)

        model_edge.train_state.step += 1
        log_utils.exp_tracker.add_scalar(Metrics.TRAIN_LOSS_MEAN,
                                         loss.cpu().item(),
                                         track_mean=False)
        loss.backward(retain_graph=False)
        if (i + 1) % loss_interval == 0:
            model_edge.optimizer.step()
            model_edge.optimizer.zero_grad(set_to_none=True)

        model_edge.train_state.accum_step += 1

        if model_edge.scheduler is not None: model_edge.scheduler.step()
        if (i + 1) % log_interval == 0:
            lr = model_edge.optimizer.param_groups[0]["lr"]
            elapsed = time.time() - start
            print(("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                   + "| Tasks / Sec: %7.1f | Learning Rate: %6.1e")
                  % (
                      i, i, loss,
                      model_edge.train_state.step / elapsed,
                      lr))

        if (i + 1) % val_interval == 0:
            settings.empty_cache()
            val_model(val_data_loader, model)
            model_node.store_metrics(
                iter_metrics={
                    Metrics.VAL_ACC: log_utils.exp_tracker.pop_scalars(
                        Metrics.VAL_ACC)
                })
    model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN,
                             log_utils.exp_tracker.get_mean_scalar(loss))
    model_edge.record_metric(Metrics.VAL_ACC,
                             log_utils.exp_tracker.get_mean_scalar(loss))
    return model_edge.complete_model_transition()


@torch.no_grad()
def val_model(
        val_data_loader: torch.utils.data.DataLoader[Sent2TagMetaTaskBatch],
        model: BaseTransformer
) -> (torch.Tensor, TrainState):
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
        n_cls, tsk_sz, _ = batch.supports.shape
        support_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.forward(
            src_ids=batch.supports.flatten(0, 1),
            tgt_ids=batch.tgt_tag_ids_supports.flatten(0, 1),
            src_mask=batch.support_masks.flatten(0, 1),
            tgt_mask=batch.tgt_tag_masks_supports.flatten(0, 1)
        )
        support_embeds: FloatTensorT[
            'NClasses,TaskSize,EmbedLen'] = \
            support_embeds.reshape(n_cls, tsk_sz, -1)
        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.forward(
            src_ids=batch.queries,
            tgt_ids=batch.tgt_tag_ids_queries,
            src_mask=batch.query_masks,
            tgt_mask=batch.tgt_tag_masks_queries
        )
        query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
        loss = torch.tensor(0.0).to(query_embeds.device)
        support_centers: FloatTensorT['NClasses,EmbedLen'] = \
            support_embeds.mean(dim=1, keepdim=False)
        distance_to_supports_per_cls: List[FloatTensorT['NQuery']] = []
        # TODO can probably do this in a batched way
        for cls in range(support_embeds.shape[0]):
            distance_to_supports_per_cls.append(
                torch.linalg.norm(
                    query_embeds - support_centers[cls, :], dim=-1, ord=2))
        distance_to_supports_per_cls: FloatTensorT['NQuery,NClasses'] = \
            FloatTensorT(torch.stack(
                distance_to_supports_per_cls, dim=-1))

        # Calculate accuracy
        predictions: LongTensorT['NQuery'] = torch.softmax(
            -1 * distance_to_supports_per_cls, dim=-1).argmax(-1)
        accuracy = (predictions == query_lbls).float().mean()
        accuracies.append(accuracy.item())
        print(f'Val Accuracy: {np.mean(accuracies)}')
    log_utils.exp_tracker.add_scalars(Metrics.VAL_ACC, accuracies,
                                      track_mean=True)
    if started_training:
        model.train()


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
              model_edge: vc.ModelTransitionEdge, device=None,
              distributed: bool = False,
              turn_off_shuffle=False,
              val_ds: BaseDataset = None, n_epochs=1) -> vc.ModelNode:
        # from mgz.ds.sentence_datasets.aus_legal_case_reports import TagGroupedSampler
        model_node.model.train()
        if model_node.mean_metrics is None:
            pass
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
            run_epoch(
                model_node,
                data_loader=train_dl,
                val_data_loader=val_dl,
                model_edge=model_edge
            )
        return model_node

    def evaluate(self, model_node: ModelNode, val_ds: BaseDataset,
                 device=None, ):
        if device is None:
            device = settings.DEVICE
        val_ds.load_validation_data()
        val_dl = val_ds.create_dataloaders(device, batch_size=1,
                                           is_distributed=False,
                                           turn_off_shuffle=True)
        val_model(val_dl, model_node.model)

    def predict(self, model_node: ModelNode):
        pass
