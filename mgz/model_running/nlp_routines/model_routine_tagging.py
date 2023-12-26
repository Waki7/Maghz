from __future__ import annotations

import time
from enum import Enum

import torch.utils.data
from tqdm import tqdm

import mgz.settings as settings
import mgz.version_control as vc
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.metrics.nlp.metrics import DistanceMeasuresPerClass
from mgz.model_running.base_routine import BaseProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class DistanceMeasure(Enum):
    L1 = 1
    COSINE = 3
    MAX_INNER_PRODUCT = 4


class TaggingRoutine(BaseProtocol):
    def __init__(self,
                 distance_measure: DistanceMeasure):
        super().__init__()
        self.distance_measure = distance_measure
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def predict_with_centers(self,
                             support_embedding: FloatTensorT[
                                 'NClasses,EmbedLen'],
                             query_embedding: FloatTensorT[
                                 'NQuery,EmbedLen']) -> \
            FloatTensorT['NQuery,NClasses']:
        similarity_to_classes: FloatTensorT['NQuery,NClasses'] = None
        if self.distance_measure == DistanceMeasure.L1:
            similarity_to_classes = -1 * DistanceMeasuresPerClass.euclidean_distance(
                class_embeddings=support_embedding,
                query_embeddings=query_embedding)
        elif self.distance_measure == DistanceMeasure.COSINE:
            similarity_to_classes = DistanceMeasuresPerClass.cosine_similarity(
                class_embeddings=support_embedding,
                query_embeddings=query_embedding)
        elif self.distance_measure == DistanceMeasure.MAX_INNER_PRODUCT:
            similarity_to_classes = DistanceMeasuresPerClass.inner_dot_product(
                class_embeddings=support_embedding,
                query_embeddings=query_embedding)
        return similarity_to_classes

    def run_prototype_encoder_decoder(self, model: EncoderDecoderTransformer,
                                      batch: Sent2TagMetaTaskBatch,
                                      gpu_max_batch_size=4) -> \
            Tuple[FloatTensorT['NQuery,NClasses'], LongTensorT['NQuery']]:
        is_encoder_decoder = isinstance(model, EncoderDecoderTransformer) and \
                             isinstance(batch, Sent2TagMetaTaskBatch)
        assert is_encoder_decoder, 'Bad combination of model and batch'

        with torch.no_grad():
            n_cls, tsk_sz, src_len = batch.supports.shape
            support = batch.supports.view(n_cls * tsk_sz, src_len)
            support_mask = batch.support_masks.view(n_cls * tsk_sz, src_len)
            support_embed_list: List[FloatTensorT['TaskSize//N,EmbedLen']] = []
            tgt_len = batch.tgt_tag_ids_supports.shape[-1]
            tgt_support = batch.tgt_tag_ids_supports.view(n_cls * tsk_sz,
                                                          tgt_len)
            tgt_mask = batch.tgt_tag_masks_supports.view(n_cls * tsk_sz,
                                                         tgt_len)
            for i in range(0, n_cls * tsk_sz, gpu_max_batch_size):
                support_embed_list.append(model.encoder_decoder_embedding(
                    src_ids=support[i:i + gpu_max_batch_size, :],
                    tgt_ids=tgt_support[i:i + gpu_max_batch_size, :],
                    src_mask=support_mask[i:i + gpu_max_batch_size, :],
                    tgt_mask=tgt_mask[i:i + gpu_max_batch_size, :]
                ))
            support_embeds: FloatTensorT['TaskSize,EmbedLen'] = \
                FloatTensorT(torch.cat(support_embed_list, dim=0))
            support_embeds: FloatTensorT[
                'NClasses,TaskSize,EmbedLen'] = \
                support_embeds.view(n_cls, tsk_sz, model.embed_dim)
            support_centers: FloatTensorT['NClasses,EmbedLen'] = \
                support_embeds.mean(dim=1, keepdim=False)
            del support_embeds
            settings.empty_cache()
        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
            src_ids=batch.queries,
            tgt_ids=batch.tgt_tag_ids_queries,
            src_mask=batch.query_masks,
            tgt_mask=batch.tgt_tag_masks_queries
        )
        query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'TaskSize,NClasses'] = self.predict_with_centers(
            support_centers, query_embeds)
        return cls_logits, query_lbls

    def run_prototype_decoder(self, model: DecoderTransformer,
                              batch: TagQAMetaTaskBatch,
                              gpu_max_batch_size=4) -> \
            Tuple[FloatTensorT['NQuery,NClasses'], LongTensorT['NQuery']]:
        is_decoder = isinstance(model, DecoderTransformer) and \
                     isinstance(batch, TagQAMetaTaskBatch)
        assert is_decoder, 'Bad combination of model and batch'

        with torch.no_grad():
            n_cls, tsk_sz, src_len = batch.supports.shape
            support = batch.supports.view(n_cls * tsk_sz, src_len)
            support_mask = batch.support_masks.view(n_cls * tsk_sz, src_len)
            support_embed_list: List[FloatTensorT['TaskSize//N,EmbedLen']] = []
            for i in range(0, n_cls * tsk_sz, gpu_max_batch_size):
                support_embed_list.append(model.decoder_embedding(
                    src_ids=support[i:i + gpu_max_batch_size, :],
                    src_mask=support_mask[i:i + gpu_max_batch_size, :],
                ))
            support_embeds: FloatTensorT['TaskSize,EmbedLen'] = \
                FloatTensorT(torch.cat(support_embed_list, dim=0))
            support_embeds: FloatTensorT[
                'NClasses,TaskSize,EmbedLen'] = \
                support_embeds.view(n_cls, tsk_sz, model.embed_dim)
            support_centers: FloatTensorT['NClasses,EmbedLen'] = \
                support_embeds.mean(dim=1, keepdim=False)
            del support_embeds
            settings.empty_cache()
        query_embeds: FloatTensorT[
            'TaskSize,EmbedLen'] = model.decoder_embedding(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
        )
        assert isinstance(model, DecoderTransformer)
        query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'TaskSize,NClasses'] = self.predict_with_centers(
            support_centers, query_embeds)
        return cls_logits, query_lbls

    def run_batch(self, model: Union[EncoderDecoderTransformer,
    DecoderTransformer],
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge,
                  gpu_max_batch_size=4) -> \
            Tuple[FloatTensorT['1'], float]:
        optimizer, scheduler = model_edge.optimizer, model_edge.scheduler
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']

        if isinstance(model, EncoderDecoderTransformer):
            cls_logits, query_lbls = self.run_prototype_encoder_decoder(model,
                                                                        batch,
                                                                        gpu_max_batch_size)
        elif isinstance(model, DecoderTransformer):
            cls_logits, query_lbls = self.run_prototype_decoder(model, batch,
                                                                gpu_max_batch_size)
        else:
            raise NotImplementedError
        loss = model_edge.loss_fn(cls_logits, query_lbls)

        # Calculate accuracy
        predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
        accuracy = (predictions == query_lbls).float().mean().item()
        model_edge.record_metric(Metrics.TRAIN_ACC_MEAN, accuracy)

        model_edge.train_state.step += 1
        model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN, loss.cpu().item())

        model_edge.train_state.accum_step += 1
        if (batch_num + 1) % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            if scheduler is not None: scheduler.step()

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

            cls_probs: ['NQuery,NClasses']
            query_lbls: LongTensorT['NQuery']

            if isinstance(model, EncoderDecoderTransformer):
                cls_probs, query_lbls = \
                    self.run_prototype_encoder_decoder(model, batch)
            elif isinstance(model, DecoderTransformer):
                cls_probs, query_lbls = self.run_prototype_decoder(model,
                                                                   batch)
            loss = model_edge.loss_fn(cls_probs, query_lbls)
            if (batch_num + 1) % gradient_accumulation_steps == 0:
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

            if (batch_num + 1) % log_interval == 0:
                lr = model_edge.optimizer.param_groups[0]["lr"]
                elapsed = time.time() - start
                print(("Epoch Step: %6d | Accumulation Step: %3d | Loss: %6.2f "
                       + "| Tasks / Sec: %7.1f | Learning Rate: %6.1e")
                      % (
                          i, i, loss,
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

            cls_probs: ['NQuery,NClasses']
            query_lbls: LongTensorT['NQuery']
            if isinstance(model, EncoderDecoderTransformer):
                cls_probs, query_lbls = \
                    self.run_prototype_encoder_decoder(model, batch)
            elif isinstance(model, DecoderTransformer):
                cls_probs, query_lbls = self.run_prototype_decoder(model,
                                                                   batch)  # Calculate accuracy
            predictions: LongTensorT['NQuery'] = cls_probs.argmax(-1)
            # batch.summarize_batch(model_node.tokenizer)
            # print('Correct Labels:', predictions)
            accuracy = (predictions == query_lbls).float().mean()
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
            self.run_epoch(
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
        self.val_model(val_dl, model_node)

    def predict(self, model_node: ModelNode):
        pass
