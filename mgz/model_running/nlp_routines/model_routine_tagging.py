from __future__ import annotations

from enum import Enum

import torch.utils.data
import transformers as hug

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.metrics.nlp.metrics import DistanceMeasuresPerClass
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
        routine: TaggingRoutine = None) -> FloatTensorT['NQuery,NClasses']:
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
        if len(no_yes_logits.shape) == 3:
            last_words = torch.argmax(no_yes_logits[:, -5:, :], dim=-1)
            if routine is not None:
                decoded = routine.tokenizer.batch_decode(last_words,
                                                         skip_special_tokens=True)
            no_yes_logits = no_yes_logits[:, -1, :]
            inference_context: InferenceContext = InferenceContext.get_llama_no_yes_scores(
                routine.tokenizer)
            similarity_to_classes = inference_context.get_word_scores_from_logits(
                no_yes_logits)
        else:
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
                 debug: bool = False):
        super().__init__(tokenizer=tokenizer,
                         gpu_max_batch_size=gpu_max_batch_size, debug=debug)
        self.distance_measure = distance_measure
        self.train_init = False
        self.eval_init = False
        self.inference_context = InferenceContext.get_llama_no_yes_scores(
            self.tokenizer)
        self.predict_init = False

    @overrides(BaseProtocol)
    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def run_prototype_decoder(self, model: DecoderTransformer,
                              batch: TagQAMetaTaskBatch) -> \
            Tuple[FloatTensorT['NQuery,NClasses'], LongTensorT['NQuery'],
            Optional[FloatTensorT['NQuery,NClasses']]]:
        is_decoder = isinstance(model, DecoderTransformer) and \
                     isinstance(batch, TagQAMetaTaskBatch)
        assert is_decoder, 'Bad combination of model and batch'

        with torch.no_grad():
            n_cls, n_sup, src_len = batch.supports.shape
            support = batch.supports.view(n_cls * n_sup, src_len)
            support_mask = batch.support_masks.view(n_cls * n_sup, src_len)
            support_embed_total: FloatTensorT[
                'n_cls,1,EmbedLen'] = FloatTensorT(
                torch.zeros((n_cls, 1, model.embed_dim))).to(support.device)

            for i in range(0, n_cls * n_sup, self.gpu_max_batch_size):
                batch_embeds, _ = model.decode_relevance(
                    src_ids=support[i:i + self.gpu_max_batch_size, :],
                    src_mask=support_mask[i:i + self.gpu_max_batch_size, :],
                    inference_context=self.inference_context
                )
                support_embed_total += batch_embeds.detach().view(n_cls, -1,
                                                                  model.embed_dim).sum(
                    dim=1, keepdim=True)
            support_centers: FloatTensorT[
                'NClasses,EmbedLen'] = (support_embed_total / n_sup).squeeze(1)
        self.debug_log_batch_info(batch)
        query_embeds: FloatTensorT['NQuery,EmbedLen']
        no_yes_logits: FloatTensorT['NQuery,VocabSize']
        query_embeds, no_yes_logits = model.decode_relevance(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
            inference_context=self.inference_context
        )
        assert isinstance(model, DecoderTransformer)
        query_lbls: LongTensorT['NQuery'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'NQuery,NClasses'] = predict_with_prototypes(
            query_embeds, support_centers, no_yes_logits, routine=self,
            distance_measure=self.distance_measure)
        return cls_logits, query_lbls, no_yes_logits

    @overrides(BaseProtocol)
    def run_batch(self,
                  model: Union[EncoderDecoderTransformer, DecoderTransformer],
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[FloatTensorT['1'], float]:
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        cls_logits, query_lbls, query_logits = self.run_prototype_decoder(
            model, batch)

        # Calculate accuracy
        with torch.no_grad():
            cls_logits_weighted = (batch.n_support_per_cls * torch.softmax(
                cls_logits.detach(), dim=-1))
            query_probs = torch.softmax(query_logits.detach(), dim=-1)
            pred_augment_weak: LongTensorT['NQuery'] = (
                    cls_logits_weighted + query_probs).argmax(-1)
            accuracy_augment_weak = (
                    pred_augment_weak == query_lbls).float().mean().item()

            pred_augment_strong: LongTensorT['NQuery'] = (
                    cls_logits_weighted + (2 * query_probs)).argmax(-1)
            accuracy_augment_strong = (
                    pred_augment_strong == query_lbls).float().mean().item()

            predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
            accuracy = (predictions == query_lbls).float().mean().item()
            if accuracy < 1.0:
                print('Accuracy is less than 1.0, predictions were')
                for i in range(len(predictions)):
                    if predictions[i] != query_lbls[i]:
                        print('----------------------------------')
                        print(
                            f'Predicted {predictions[i]} for label {query_lbls[i]}')
                        print(f'query: {self.tokenizer.decode(batch.queries[i])}')
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
