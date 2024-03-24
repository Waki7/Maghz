from __future__ import annotations

import torch.utils.data
import transformers as hug

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.sequence_to_sequence_datasets.behavioral_batch import \
    BehavioralBatch
from mgz.model_running.base_routine import BaseProtocol
from mgz.model_running.nlp_routines.base_nlp_routine import BaseNLPProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer, ModelType
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


class TaggingRoutine(BaseNLPProtocol):
    def __init__(self,
                 tokenizer: hug.PreTrainedTokenizerBase = None,
                 gpu_max_batch_size: int = 4,
                 gradient_accumulation_steps: int = 4,
                 debug: bool = False, ):
        super().__init__(tokenizer=tokenizer,
                         gpu_max_batch_size=gpu_max_batch_size, debug=debug,
                         gradient_accumulation_steps=gradient_accumulation_steps)
        self.train_init = False
        self.eval_init = False
        self.predict_init = False

    @overrides(BaseProtocol)
    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    def run_prototype_decoder(self, model: DecoderTransformer,
                              batch: BehavioralBatch) -> \
            FloatTensorT['B,TgtSeqLen,EmbedLen']:
        is_decoder = model.MODEL_TYPE == ModelType.DecoderTransformer and \
                     isinstance(batch, BehavioralBatch)
        assert is_decoder, 'Bad combination of model and batch'
        logits: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = model.forward(
            src_ids=batch.src_ids, src_mask=batch.src_masks)
        return logits

    @overrides(BaseProtocol)
    def run_batch(self,
                  model: Union[EncoderDecoderTransformer, DecoderTransformer],
                  batch: BehavioralBatch,
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[FloatTensorT['1'], float]:
        cls_logits: ['NQuery,NClasses']
        query_lbls: LongTensorT['NQuery']
        no_yes_logits: FloatTensorT['NQuery,NClasses']
        cls_logits, query_lbls, no_yes_logits = self.run_prototype_decoder(
            model, batch)

        # Calculate accuracy
        with torch.no_grad():
            pred_classification = no_yes_logits.argmax(-1)
            classification_accuracy = (
                    pred_classification == query_lbls).float().mean().item()

            predictions: LongTensorT['NQuery'] = cls_logits.argmax(-1)
            accuracy = (predictions == query_lbls).float().mean().item()

        if model_edge is not None:
            total_samples = cls_logits.shape[0] * cls_logits.shape[1]

            no_yes_log_probs = torch.log_softmax(no_yes_logits, dim=-1)
            prototypical_probs = torch.log_softmax(cls_logits, dim=-1)
            loss = model_edge.loss_fn(cls_logits, query_lbls)
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

                model_edge.log_metric("train/accuracy_classification",
                                      classification_accuracy)

            else:
                model_edge.record_metric(Metrics.VAL_AVG_PRED,
                                         predictions.float().mean().item())

                model_edge.log_metric("val/accuracy_all_classification",
                                      classification_accuracy)



        else:
            loss = FloatTensorT([0.0])
        return loss, accuracy

    @overrides(BaseProtocol)
    def predict(self, model_node: ModelNode):
        pass
