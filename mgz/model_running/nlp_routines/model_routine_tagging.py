from __future__ import annotations

import time

import torch.utils.data
from accelerate import Accelerator
from tqdm import tqdm

import mgz.settings as settings
import mgz.version_control as vc
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.models.nlp.base_transformer import BaseTransformer, \
    EncoderDecoderTransformer, DecoderTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


def predict_with_centers(
        support_embedding: FloatTensorT['NClasses,TaskSize,EmbedLen'],
        query_embedding: FloatTensorT['TaskSize,EmbedLen']) -> \
        ProbTensorT['NQuery,NClasses']:
    distance_to_supports_per_cls: FloatTensorT['NQuery,NClasses'] = \
        torch.linalg.norm(
            query_embedding[:, None, :] - support_embedding[None, :, :], dim=-1,
            ord=2)
    cls_probs = ProbTensorT(torch.softmax(
        -1 * distance_to_supports_per_cls, dim=-1))
    return cls_probs


def run_prototype(model: BaseTransformer, batch: TagQAMetaTaskBatch) -> \
        Tuple[ProbTensorT['NQuery,NClasses'], LongTensorT['NQuery']]:
    with torch.no_grad():
        n_cls, tsk_sz, src_len = batch.supports.shape
        if isinstance(batch, Sent2TagMetaTaskBatch) and isinstance(model,
                                                                   EncoderDecoderTransformer):
            n_cls, tsk_sz, tgt_len = batch.tgt_tag_ids_supports.shape
            support_embeds: FloatTensorT[
                'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
                src_ids=batch.supports.view(n_cls * tsk_sz, src_len),
                tgt_ids=batch.tgt_tag_ids_supports.view(n_cls * tsk_sz,
                                                        tgt_len),
                src_mask=batch.support_masks.flatten(0, 1),
                tgt_mask=batch.tgt_tag_masks_supports.flatten(0, 1)
            )
        elif isinstance(batch, TagQAMetaTaskBatch) and isinstance(model,
                                                                  DecoderTransformer):
            support_embeds: FloatTensorT[
                'TaskSize,EmbedLen'] = model.decoder_embedding(
                src_ids=batch.supports.view(n_cls * tsk_sz, src_len),
                src_mask=batch.support_masks.flatten(0, 1),
            )
        else:
            raise ValueError('Bad combination of batch and model')
        support_embeds: FloatTensorT[
            'NClasses,TaskSize,EmbedLen'] = \
            support_embeds.view(n_cls, tsk_sz, -1)
        support_centers: FloatTensorT['NClasses,EmbedLen'] = \
            support_embeds.mean(dim=1, keepdim=False)
        del support_embeds
        settings.empty_cache()
    if isinstance(batch, Sent2TagMetaTaskBatch) and isinstance(model,
                                                               EncoderDecoderTransformer):
        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
            src_ids=batch.queries,
            tgt_ids=batch.tgt_tag_ids_queries,
            src_mask=batch.query_masks,
            tgt_mask=batch.tgt_tag_masks_queries
        )
    elif isinstance(batch, TagQAMetaTaskBatch) and isinstance(model,
                                                              DecoderTransformer):
        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.decoder_embedding(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
        )

    else:
        raise ValueError('Bad combination of batch and model')
    assert isinstance(model, DecoderTransformer)
    query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
    cls_probs: ProbTensorT['TaskSize,NClasses'] = predict_with_centers(
        support_centers, query_embeds)
    return cls_probs, query_lbls


def run_epoch(
        model_node: ModelNode,
        data_loader: torch.utils.data.DataLoader[TagQAMetaTaskBatch],
        val_data_loader: torch.utils.data.DataLoader[TagQAMetaTaskBatch],
        model_edge: ModelTransitionEdge,
        log_interval=5,
        val_interval=50,
) -> ModelNode:
    """Train a single epoch"""
    model, tokenizer = model_node.model, model_node.tokenizer
    model.train()

    start = time.time()
    i: int
    accelerator = Accelerator(gradient_accumulation_steps=4)

    batch: Sent2TagMetaTaskBatch
    if not model_node.has_initial_metrics():
        model_node.store_metrics(val_model(val_data_loader, model, model_edge))
    model, optimizer, training_dataloader, scheduler = (
        accelerator.prepare(model, model_edge.optimizer, data_loader,
                            model_edge.scheduler))

    for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
        if batch is None:
            logging.warning(
                'batch is None... need to fix issue of tags are super sparse')
            continue

        cls_probs: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        with accelerator.accumulate(model):
            cls_probs, query_lbls = run_prototype(model, batch)
            loss = torch.tensor(0.0).to(query_lbls.device)
            loss += model_edge.loss_fn(cls_probs, query_lbls)
            accelerator.backward(loss)
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()

        # Calculate accuracy
        predictions: LongTensorT[
            'NQuery'] = cls_probs.argmax(-1)
        accuracy = (predictions == query_lbls).float().mean().item()
        model_edge.record_metric(Metrics.TRAIN_ACC_MEAN, accuracy)

        model_edge.train_state.step += 1
        model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN, loss.cpu().item())

        model_edge.train_state.accum_step += 1

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
            val_model(val_data_loader, model, model_edge)
        settings.empty_cache()
        del batch
    return model_edge.complete_model_transition()


@torch.no_grad()
def val_model(
        val_data_loader: torch.utils.data.DataLoader[TagQAMetaTaskBatch],
        model: BaseTransformer, training_edge: ModelTransitionEdge = None
) -> Dict[Metrics, Union[float, List[float]]]:
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

        cls_probs: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        cls_probs, query_lbls = run_prototype(model, batch)
        # Calculate accuracy
        predictions: LongTensorT['NQuery'] = cls_probs.argmax(-1)
        accuracy = (predictions == query_lbls).float().mean()
        accuracies.append(accuracy.item())
        print(f'Val Accuracy Moving Avg: {np.mean(accuracies)}')
    if training_edge is not None:
        training_edge.record_validation(accuracies)
    if started_training:
        model.train()
    return {
        Metrics.VAL_ACC_ALL: accuracies,
        Metrics.VAL_ACC_MEAN: np.mean(accuracies)
    }


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
