from __future__ import annotations

from enum import Enum

import torch.utils.data
import transformers as hug

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.responsivenes_datasets.responsive_batch import \
    TagQAMetaTaskBatch
from mgz.math_utils.nlp.metrics import DistanceMeasuresPerClass
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.nlp_routines.base_nlp_routine import BaseNLPProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer, InferenceContext, ModelType
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
            query_embeddings=query_embedding, normalize=True)
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

        no_yes_weight = (int(0.5 * n_supports) + 2)
        no_yes_probs_weighted = no_yes_weight * no_yes_probs

        proto_weight = n_supports
        query_probs_weighted = proto_weight * query_probs

        pred_augment_strong: ProbTensorT['NQuery,NClasses'] = \
            (query_probs_weighted + no_yes_probs_weighted) / (
                    proto_weight + no_yes_weight)
        return pred_augment_strong
    else:
        return no_yes_probs


class TaggingRoutine(BaseNLPProtocol):
    def __init__(self,
                 tokenizer: hug.PreTrainedTokenizerBase = None,
                 gpu_max_batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 debug: bool = False,
                 distance_measure: DistanceMeasure = None, ):
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
        is_decoder = model.MODEL_TYPE == ModelType.DecoderTransformer and \
                     isinstance(batch, TagQAMetaTaskBatch)
        assert is_decoder, 'Bad combination of model and batch'

        with torch.no_grad():
            support_centers: FloatTensorT[
                'NClasses,EmbedLen'] = self._2dim_batchify(
                batch, model.decode_embedding_w_lm_logits, model.embed_dim,
                self.gpu_max_batch_size)

        self.debug_log_batch_info(batch)
        query_embeds: FloatTensorT['NQuery,EmbedLen']
        no_yes_logits: FloatTensorT['NQuery,VocabSize']
        query_embeds, lm_logits = model.decode_embedding_w_lm_logits(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
        )
        no_yes_logits = (
            InferenceContext(self.tokenizer).get_word_scores_from_logits(
                lm_logits))
        query_lbls: LongTensorT['NQuery'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'NQuery,NClasses'] = predict_with_prototypes(
            query_embeds, support_centers, no_yes_logits, routine=self,
            distance_measure=self.distance_measure)

        return cls_logits, query_lbls, no_yes_logits

    @overrides(BaseProtocol)
    def run_batch(self,
                  model: Union[EncoderDecoderTransformer, DecoderTransformer],
                  batch: TagQAMetaTaskBatch,
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[FloatTensorT['1'], float]:
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        no_yes_logits: FloatTensorT['NQuery,NClasses']
        cls_logits, query_lbls, no_yes_logits = self.run_prototype_decoder(
            model, batch)

        if self.distance_measure == DistanceMeasure.L2:
            softmax_temp = 10.0
        else:
            softmax_temp = 1.0

        # Calculate accuracy
        with torch.no_grad():
            proto_probs = torch.softmax(cls_logits.detach() / softmax_temp,
                                        dim=-1)
            no_yes_probs = torch.softmax(no_yes_logits.detach(), dim=-1)

            proto_probs_weighted = (batch.n_support_per_cls * proto_probs)

            pred_augment_weak: LongTensorT['NQuery'] = (
                    proto_probs_weighted + no_yes_probs).argmax(-1)
            accuracy_augment_weak = (
                    pred_augment_weak == query_lbls).float().mean().item()

            pred_augment_strong: LongTensorT['NQuery'] = (
                    proto_probs_weighted + (2 * no_yes_probs)).argmax(-1)
            accuracy_augment_strong = (
                    pred_augment_strong == query_lbls).float().mean().item()

            pred_classification = no_yes_logits.argmax(-1)
            classification_accuracy = (
                    pred_classification == query_lbls).float().mean().item()

            predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
            accuracy = (predictions == query_lbls).float().mean().item()

        if model_edge is not None:
            total_samples = cls_logits.shape[0] * cls_logits.shape[1]

            no_yes_log_probs = torch.log_softmax(no_yes_logits, dim=-1)
            noisy_no_yes_probs = batch.use_heuristic_to_identify_hard_query(
                no_yes_log_probs=no_yes_log_probs.detach(),
                tokenizer=self.tokenizer)
            prototypical_probs = torch.log_softmax(cls_logits / softmax_temp,
                                                   dim=-1)
            loss = model_edge.loss_fn(
                prototypical_probs + noisy_no_yes_probs.detach(),
                query_lbls)
            # print('loss1', loss)
            if classification_accuracy < 1.0:
                loss += 0.5 * (
                    model_edge.loss_fn(no_yes_log_probs, query_lbls))
                # print('loss', loss)
            # print('total_samples', total_samples)
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
                model_edge.log_metric("train/accuracy_classification",
                                      classification_accuracy)

                model_edge.log_metric("train/proto_score_no",
                                      proto_probs[:, 0].mean(0))
                model_edge.log_metric("train/proto_score_yes",
                                      proto_probs[:, 1].mean(0))
                model_edge.log_metric("train/noyes_score_no",
                                      no_yes_probs[:, 0].mean(0))
                model_edge.log_metric("train/noyes_score_yes",
                                      no_yes_probs[:, 1].mean(0))

            else:
                model_edge.record_metric(Metrics.VAL_AVG_PRED,
                                         predictions.float().mean().item())
                model_edge.log_metric("val/accuracy_all_augment_weak",
                                      accuracy_augment_weak)
                model_edge.log_metric("val/accuracy_all_augment_strong",
                                      accuracy_augment_strong)
                model_edge.log_metric("val/accuracy_all_augment_strong",
                                      accuracy_augment_strong)
                model_edge.log_metric("val/accuracy_all_classification",
                                      classification_accuracy)

                model_edge.log_metric("val/proto_score_no",
                                      proto_probs[:, 0].mean(0))
                model_edge.log_metric("val/proto_score_yes",
                                      proto_probs[:, 1].mean(0))
                model_edge.log_metric("val/noyes_score_no",
                                      no_yes_probs[:, 0].mean(0))
                model_edge.log_metric("val/noyes_score_yes",
                                      no_yes_probs[:, 1].mean(0))


        else:
            loss = FloatTensorT([0.0])
        return loss, accuracy

    @overrides(BaseProtocol)
    def predict(self, model_node: ModelNode):
        pass
