from __future__ import annotations

from enum import Enum

import torch.utils.data
import transformers as hug

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.math_utils.nlp.metrics import DistanceMeasuresPerClass
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.nlp_routines.base_nlp_routine import BaseNLPProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer, InferenceContext
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class DistanceMeasure(Enum):
    L2 = 1
    COSINE = 2
    MAX_INNER_PRODUCT = 3
    CLASSIFICATION = 4
    WEIGHTED_HYBRID = 5


def predict_with_prototypes(
        query_embedding: FloatTensorT['NQuery,EmbedLen'],
        support_embedding: FloatTensorT['NClasses,EmbedLen'],
        no_yes_logits: FloatTensorT['NQuery,VocabSize'] = None,
        distance_measure: DistanceMeasure = None,
        routine: TaggingRoutine = None) -> FloatTensorT[
    'NQuery,NClasses']:
    similarity_to_classes: FloatTensorT['NQuery,NClasses']
    if distance_measure == DistanceMeasure.L2:
        similarity_to_classes = -1 * DistanceMeasuresPerClass.euclidean_distance(
            class_embeddings=support_embedding,
            query_embeddings=query_embedding)
    elif distance_measure == DistanceMeasure.COSINE:
        similarity_to_classes = DistanceMeasuresPerClass.cosine_similarity(
            class_embeddings=support_embedding,
            query_embeddings=query_embedding)
    elif distance_measure == DistanceMeasure.MAX_INNER_PRODUCT:
        similarity_to_classes = DistanceMeasuresPerClass.inner_dot_product(
            class_embeddings=support_embedding,
            query_embeddings=query_embedding)
    elif distance_measure == DistanceMeasure.CLASSIFICATION:
        similarity_to_classes = no_yes_logits
        routine.debug_log_predictions(similarity_to_classes)
    else:
        raise ValueError(f'Not supported distance measure {distance_measure}')
    return similarity_to_classes


def predict_probs_with_optional_prototypes(
        query_embedding: FloatTensorT['NQuery,EmbedLen'],
        no_yes_probs: ProbTensorT['NQuery,VocabSize'],
        support_embedding: FloatTensorT['NClasses,EmbedLen'] = None,
        n_supports: int = None) -> ProbTensorT['NQuery,NClasses']:
    if support_embedding is not None:
        assert n_supports is not None
        similarity_to_classes: FloatTensorT['NQuery,NClasses'] = (
                -1 * DistanceMeasuresPerClass.euclidean_distance(
            class_embeddings=support_embedding,
            query_embeddings=query_embedding))
        query_probs = torch.softmax(similarity_to_classes, dim=-1)
        cls_probs_weighted = (query_probs)
        pred_augment_weak: ProbTensorT['NQuery,NClasses'] = (
                (cls_probs_weighted + no_yes_probs) / (2))
        return pred_augment_weak
    else:
        return no_yes_probs


class TaggingRoutine(BaseNLPProtocol):
    def __init__(self,
                 distance_measure: DistanceMeasure = None,
                 tokenizer: hug.PreTrainedTokenizerBase = None,
                 gpu_max_batch_size: int = 4,
                 debug: bool = False,
                 gradient_accumulation_steps: int = 4, ):
        super().__init__(tokenizer=tokenizer,
                         gpu_max_batch_size=gpu_max_batch_size, debug=debug,
                         gradient_accumulation_steps=gradient_accumulation_steps)
        self.distance_measure = distance_measure
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    @overrides(BaseProtocol)
    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def _2dim_batchify(self, batch: TagQAMetaTaskBatch,
                       decode_function,
                       embed_dim: int,
                       gpu_max_batch_size: int) -> \
            FloatTensorT['NClasses,EmbedLen']:
        n_cls, n_sup, src_len = batch.supports.shape
        support: LongTensorT[
            'NClasses,NSupport/NClasses,SrcSeqLen'] = batch.supports
        support_mask: LongTensorT[
            'NClasses,NSupport/NClasses,SrcSeqLen'] = batch.support_masks
        support_embed_total: FloatTensorT[
            'n_cls,1,EmbedLen'] = FloatTensorT(
            torch.zeros((n_cls, 1, embed_dim))).to(support.device)

        support_ids: LongTensorT[
            'n_cls,NSupport/NClasses,EmbedLen'] = FloatTensorT(
            torch.zeros((n_cls, n_sup, src_len))).to(support.device)
        for i in range(0, n_cls):
            for j in range(0, n_sup, gpu_max_batch_size):
                end = min(j + gpu_max_batch_size, n_sup)
                batch_embeds: FloatTensorT['n_sup,EmbedLen']
                batch_embeds, logits = decode_function(
                    src_ids=support[i, j: end, :],
                    src_mask=support_mask[i, j: end, :],
                )
                support_embed_total[i] += batch_embeds.detach().sum(dim=0,
                                                                    keepdim=True)
                support_ids[i, j: end, :] = support[i, j: end, :]
        support_centers: FloatTensorT[
            'NClasses,EmbedLen'] = (support_embed_total / n_sup).squeeze(1)
        # for cls in range(0, n_cls):
        #     print('-')
        #     print(support_ids[cls].shape)
        #     print([self.tokenizer.decode(sup, skip_special_tokens=True)[200:600]
        #            for sup in support_ids[cls]])
        # exit(3)
        return support_centers

    def run_prototype_decoder(self, model: DecoderTransformer,
                              batch: TagQAMetaTaskBatch) -> \
            Tuple[FloatTensorT['NQuery,NClasses'], LongTensorT['NQuery'],
            Optional[FloatTensorT['NQuery,NClasses']]]:
        is_decoder = isinstance(model, DecoderTransformer) and \
                     isinstance(batch, TagQAMetaTaskBatch)
        assert is_decoder, 'Bad combination of model and batch'

        with torch.no_grad():
            support_centers: FloatTensorT[
                'NClasses,EmbedLen'] = self._2dim_batchify(
                batch, model.decode_relevance, model.embed_dim,
                self.gpu_max_batch_size)

        self.debug_log_batch_info(batch)
        query_embeds: FloatTensorT['NQuery,EmbedLen']
        no_yes_logits: FloatTensorT['NQuery,VocabSize']
        query_embeds, lm_logits = model.decode_relevance(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
        )
        no_yes_logits = (
            InferenceContext(self.tokenizer).get_word_scores_from_logits(
                lm_logits))
        assert isinstance(model, DecoderTransformer)
        query_lbls: LongTensorT['NQuery'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'NQuery,NClasses'] = predict_with_prototypes(
            query_embeds, support_centers, no_yes_logits, routine=self,
            distance_measure=self.distance_measure)

        # predictions = cls_logits.argmax(-1)
        #
        # for i in range(len(predictions)):
        #     if predictions[i] != query_lbls[i]:
        #         print('----------------------------------')
        #         print('----------------------------------')
        #         print('----------------------------------')
        #         print('query_lbls', query_lbls.shape)
        #         print('predictions', predictions.shape)
        #         print('cls_logits', cls_logits.shape)
        #         InferenceContext(self.tokenizer).debug(
        #             lm_logits[i])
        #         print(no_yes_logits[i])
        #         print(
        #             f'Predicted {predictions[i]} for label {query_lbls[i]}')
        #         print('argmax', lm_logits[i].argmax(-1))
        #         print(
        #             f'query output: {self.tokenizer.decode(lm_logits[i].argmax(-1))}')
        #
        #         print(
        #             f'query input: {self.tokenizer.decode(batch.queries[i], skip_special_tokens=True)}')
        #         exit(3)
        return cls_logits, query_lbls, no_yes_logits

    @overrides(BaseProtocol)
    def run_batch(self,
                  model: Union[EncoderDecoderTransformer, DecoderTransformer],
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[FloatTensorT['1'], float]:
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        cls_logits, query_lbls, no_yes_logits = self.run_prototype_decoder(
            model, batch)

        # Calculate accuracy
        with torch.no_grad():
            cls_logits_weighted = (batch.n_support_per_cls * torch.softmax(
                cls_logits.detach(), dim=-1))
            no_yes_probs = torch.softmax(no_yes_logits.detach(), dim=-1)
            pred_augment_weak: LongTensorT['NQuery'] = (
                    cls_logits_weighted + no_yes_probs).argmax(-1)
            accuracy_augment_weak = (
                    pred_augment_weak == query_lbls).float().mean().item()

            pred_augment_strong: LongTensorT['NQuery'] = (
                    cls_logits_weighted + (2 * no_yes_probs)).argmax(-1)
            accuracy_augment_strong = (
                    pred_augment_strong == query_lbls).float().mean().item()

            predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
            accuracy = (predictions == query_lbls).float().mean().item()
            # print('no_yes_probs', no_yes_probs)
            # print('cls_logits', cls_logits)
            # print('argmax', cls_logits.argmax(-1))
            # print('query_lbls', query_lbls)
            # print('accuracy', accuracy)
            # print('accuracy_cls', (no_yes_logits.argmax(
            #     -1) == query_lbls).float().mean().item())
            # print('accuracy_augment_weak', accuracy_augment_weak)
            # print('accuracy_augment_strong', accuracy_augment_strong)
        if model_edge is not None:
            total_samples = cls_logits.shape[0] * cls_logits.shape[1]
            loss = model_edge.loss_fn(cls_logits, query_lbls)
            loss = loss / total_samples
            if model.training:
                model_edge.record_metric(Metrics.TRAIN_ACC_MEAN, accuracy)
                model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN,
                                         loss.cpu().item())
                model_edge.record_metric(Metrics.TRAIN_AVG_PRED,
                                         predictions.float().mean().item())
                model_edge.log_metric("train/accuracy_augment_weak",
                                      accuracy_augment_weak)
                model_edge.log_metric("train/accuracy_augment_strong",
                                      accuracy_augment_strong)
            else:
                model_edge.record_metric(Metrics.VAL_AVG_PRED,
                                         predictions.float().mean().item())
                model_edge.log_metric("val/accuracy_all_augment_weak",
                                      accuracy_augment_weak)
                model_edge.log_metric("val/accuracy_all_augment_strong",
                                      accuracy_augment_strong)
                model_edge.log_metric("val/accuracy_all_augment_strong",
                                      accuracy_augment_strong)
        else:
            loss = FloatTensorT([0.0])
        return loss, accuracy

    @overrides(BaseProtocol)
    def predict(self, model_node: ModelNode):
        pass
