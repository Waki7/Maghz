from __future__ import annotations

from enum import Enum

import torch.utils.data
import transformers as hug

import mgz.settings as settings
import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sentence_datasets import Sent2TagMetaTaskBatch, \
    TagQAMetaTaskBatch
from mgz.metrics.nlp.metrics import DistanceMeasuresPerClass
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.nlp_routines.base_nlp_routine import BaseNLPProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class DistanceMeasure(Enum):
    L2 = 1
    COSINE = 2
    MAX_INNER_PRODUCT = 3
    CLASSIFICATION = 4


def get_llama_no_yes_scores(logits: FloatTensorT['NQuery,NClasses']):
    assert logits.shape[-1] == 32002
    n_yes = 4
    Yes_id = 5613
    block_Yes_id = 5592
    yes_id = 9780
    block_yes_id = 5081
    yes_ids = [Yes_id, block_Yes_id, yes_id, block_yes_id]

    n_no = 6
    NO_id = 4032
    block_NO_id = 7929
    no_id = 1510
    block_no_id = 708
    No_id = 2501
    block_No_id = 1770
    no_ids = [NO_id, block_NO_id, no_id, block_no_id, No_id,
              block_No_id]

    yes_no_logits = logits[:, no_ids + yes_ids]
    no_score = torch.max(yes_no_logits[:, :n_no], dim=-1)[0]
    yes_score = torch.max(yes_no_logits[:, -n_yes:], dim=-1)[0]
    similarity_to_classes = FloatTensorT(
        torch.stack([no_score, yes_score], dim=-1))
    return similarity_to_classes
# def predict_with_centers(support_embedding: FloatTensorT[
#                              'NClasses,EmbedLen'],
#                          query_embedding: FloatTensorT[
#                              'NQuery,EmbedLen'],
#                          distance_measure: DistanceMeasure,
#                          query_logits: FloatTensorT[
#                              'NQuery,VocabSize'] = None,
#                          tokenizer = None) -> \
#         FloatTensorT['NQuery,NClasses']:
#     similarity_to_classes: FloatTensorT['NQuery,NClasses'] = None
#     if distance_measure == DistanceMeasure.L2:
#         similarity_to_classes = -1 * DistanceMeasuresPerClass.euclidean_distance(
#             class_embeddings=support_embedding,
#             query_embeddings=query_embedding)
#     elif distance_measure == DistanceMeasure.COSINE:
#         similarity_to_classes = DistanceMeasuresPerClass.cosine_similarity(
#             class_embeddings=support_embedding,
#             query_embeddings=query_embedding)
#     elif distance_measure == DistanceMeasure.MAX_INNER_PRODUCT:
#         similarity_to_classes = DistanceMeasuresPerClass.inner_dot_product(
#             class_embeddings=support_embedding,
#             query_embeddings=query_embedding)
#     elif distance_measure == DistanceMeasure.CLASSIFICATION:
#         assert query_logits.shape[-1] == 32002
#         if len(query_logits.shape) == 3:
#             last_words = torch.argmax(query_logits[:, -5:, :], dim=-1)
#             decoded = self.tokenizer.batch_decode(last_words,
#                                                   skip_special_tokens=True)
#             print('decoded ', decoded)
#             query_logits = query_logits[:, -1, :]
#         similarity_to_classes = get_llama_no_yes_scores(query_logits)
#         print('similarity_to_classes ', similarity_to_classes)
#         self.debug_log_predictions(similarity_to_classes)
#
#     return similarity_to_classes

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
        self.predict_init = False

    @overrides(BaseProtocol)
    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def predict_with_centers(self,
                             support_embedding: FloatTensorT[
                                 'NClasses,EmbedLen'],
                             query_embedding: FloatTensorT[
                                 'NQuery,EmbedLen'],
                             query_logits: FloatTensorT[
                                 'NQuery,VocabSize'] = None) -> \
            FloatTensorT['NQuery,NClasses']:
        similarity_to_classes: FloatTensorT['NQuery,NClasses'] = None
        if self.distance_measure == DistanceMeasure.L2:
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
        elif self.distance_measure == DistanceMeasure.CLASSIFICATION:
            assert query_logits.shape[-1] == 32002
            if len(query_logits.shape) == 3:
                last_words = torch.argmax(query_logits[:, -5:, :], dim=-1)
                decoded = self.tokenizer.batch_decode(last_words,
                                                      skip_special_tokens=True)
                print('decoded ', decoded)
                query_logits = query_logits[:, -1, :]
            similarity_to_classes = get_llama_no_yes_scores(query_logits)
            print('similarity_to_classes ', similarity_to_classes)
            self.debug_log_predictions(similarity_to_classes)

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
                batch_embeds = model.decode_relevance(
                    src_ids=support[i:i + gpu_max_batch_size, :],
                    src_mask=support_mask[i:i + gpu_max_batch_size, :],
                )
                support_embed_list.append(batch_embeds)
            support_embeds: FloatTensorT['TaskSize,EmbedLen'] = \
                FloatTensorT(torch.cat(support_embed_list, dim=0))
            support_embeds: FloatTensorT[
                'NClasses,TaskSize,EmbedLen'] = \
                support_embeds.view(n_cls, tsk_sz,
                                    model.embed_dim)  # 32002 model.embed_dim)
            support_centers: FloatTensorT['NClasses,EmbedLen'] = \
                support_embeds.mean(dim=1, keepdim=False)
            del support_embeds
            settings.empty_cache()
        self.debug_log_batch_info(batch)
        query_embeds: FloatTensorT['TaskSize,EmbedLen']
        query_logits: FloatTensorT['TaskSize,VocabSize'] = None
        query_embeds = model.decode_relevance(
            src_ids=batch.queries,
            src_mask=batch.query_masks,
        )
        # n_query = batch.queries.shape[0]
        # query_embed_list: List[FloatTensorT['TaskSize//N,EmbedLen']] = []
        # for i in range(0, n_query, 1):
        #     query_embed_list.append(model.decoder_embedding(
        #         src_ids=batch.queries[i:i + 1, :],
        #         src_mask=batch.query_masks[i:i + 1, :],
        #     ))
        # query_embeds: FloatTensorT['TaskSize,EmbedLen'] = \
        #     FloatTensorT(torch.cat(query_embed_list, dim=0))

        assert isinstance(model, DecoderTransformer)
        query_lbls: LongTensorT['TaskSize'] = batch.query_lbls
        cls_logits: FloatTensorT[
            'TaskSize,NClasses'] = self.predict_with_centers(
            support_centers, query_embeds, query_logits)
        return cls_logits, query_lbls

    @overrides(BaseProtocol)
    def run_batch(self,
                  model: Union[EncoderDecoderTransformer, DecoderTransformer],
                  batch: Union[Sent2TagMetaTaskBatch, TagQAMetaTaskBatch],
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[FloatTensorT['1'], float]:
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']

        if isinstance(model, EncoderDecoderTransformer):
            cls_logits, query_lbls = self.run_prototype_encoder_decoder(model,
                                                                        batch,
                                                                        self.gpu_max_batch_size)
        elif isinstance(model, DecoderTransformer):
            cls_logits, query_lbls = self.run_prototype_decoder(model, batch,
                                                                self.gpu_max_batch_size)
        else:
            raise NotImplementedError

        # Calculate accuracy
        predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
        accuracy = (predictions == query_lbls).float().mean().item()
        if model_edge is not None:
            loss = model_edge.loss_fn(cls_logits, query_lbls)
            model_edge.record_metric(Metrics.TRAIN_ACC_MEAN, accuracy)
            model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN, loss.cpu().item())
            model_edge.record_metric(Metrics.TRAIN_AVG_PRED,
                                     predictions.float().mean().item())
        else:
            loss = FloatTensorT([0.0])
        return loss, accuracy

    @overrides(BaseProtocol)
    def predict(self, model_node: ModelNode):
        pass
