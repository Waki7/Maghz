from __future__ import annotations

from einops import repeat

import spaces as sp
from mgz.ds.base_dataset import BaseDataset
from mgz.ds.sentence_datasets.datasets_reinforcement.reinforcement_batch import \
    ReinforcementBatch
from mgz.model_running.nlp_routines.base_nlp_routine import BaseNLPProtocol
from mgz.models.nlp.base_transformer import EncoderDecoderTransformer, \
    DecoderTransformer, ModelType
from mgz.typing import *

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge

import torch.utils.data
import transformers as hug
from accelerate.utils import set_seed
from tqdm import tqdm

import mgz.settings as settings
from mgz.model_running.base_routine import BaseProtocol
from mgz.typing import *
from mgz.version_control.metrics import Metrics

if TYPE_CHECKING:
    from mgz.version_control import ModelNode, ModelTransitionEdge


def log(t, eps=1e-20):
    return torch.log(t.clamp(min=eps))


def gumbel_noise(t):
    noise = torch.zeros_like(t).uniform_(0, 1)
    return -log(-log(noise))


def gumbel_sample(t, temperature=1., dim=-1):
    return ((t / max(temperature, 1e-10)) + gumbel_noise(t)).argmax(dim=dim)


class ReinforcingRoutine(BaseNLPProtocol):
    def __init__(self,
                 tokenizer: hug.PreTrainedTokenizerBase = None,
                 gpu_max_batch_size: int = 4,
                 ppo_epochs: int = 4,
                 ppo_clip: float = 0.2,
                 debug: bool = False, ):
        super().__init__(tokenizer=tokenizer,
                         gpu_max_batch_size=gpu_max_batch_size, debug=debug,
                         gradient_accumulation_steps=1)
        self.train_init = False
        self.eval_init = False
        self.predict_init = False
        self.ppo_epochs = ppo_epochs
        self.ppo_clip = ppo_clip

    @overrides(BaseProtocol)
    def _check(self, ds: BaseDataset):
        assert isinstance(ds.input_space, sp.Sentence)
        assert isinstance(ds.target_space, sp.Tagging)

    @overrides(BaseNLPProtocol)
    def run_batch(self,
                  model: Union[
                      EncoderDecoderTransformer, DecoderTransformer],
                  batch: ReinforcementBatch,
                  model_edge: ModelTransitionEdge, ) -> \
            Tuple[LogitsTensorT['B,TgtSeqLen,VocabSize'],
            LogitsTensorT['B,TgtSeqLen'],
            LongTensorT['B,TgtSeqLen'],
            FloatTensorT['B']]:
        cls_logits: FloatTensorT['B,VocabSize']
        rewards: FloatTensorT['B']

        is_decoder = model.MODEL_TYPE == ModelType.DecoderTransformer and \
                     isinstance(batch, ReinforcementBatch)
        assert is_decoder, 'Bad combination of model and batch'
        src_ids = (
            repeat(batch.src_ids, 'B SrcSeqLen -> (B P) SrcSeqLen', P=2))
        src_masks = (
            repeat(batch.src_masks, 'B SrcSeqLen -> (B P) SrcSeqLen', P=2))

        all_logits: LogitsTensorT['B,TgtSeqLen,VocabSize']
        selected_logits: LogitsTensorT['B,TgtSeqLen']
        selected_tokens: LongTensorT['B,TgtSeqLen']
        all_logits, selected_tokens, selected_logits = model.generate_logits(
            src_ids=src_ids, src_mask=src_masks, max_new_tokens=1)
        rewards: FloatTensorT['B'] = batch.get_reward(selected_tokens,
                                                      self.tokenizer)
        predictions = self.tokenizer.batch_decode(selected_tokens)
        print('predictions', predictions)
        return all_logits, selected_logits, selected_tokens, rewards

    def run_batch_with_update(self,
                              model: Union[
                                  EncoderDecoderTransformer, DecoderTransformer],
                              batch: ReinforcementBatch,
                              model_edge: ModelTransitionEdge,
                              optimizer, scheduler,
                              ):

        if model_edge is not None:
            old_action_probs = None

            actor_loss = FloatTensorT([0.0])
            for i in range(self.ppo_epochs):
                all_logits, selected_logits, selected_tokens, rewards = self.run_batch(
                    model, batch,
                    model_edge)
                log_probs = torch.nn.functional.log_softmax(all_logits, dim=-1)
                new_action_probs = log_probs.gather(-1,
                                                    selected_tokens.unsqueeze(
                                                        -1))
                if old_action_probs is None:
                    old_action_probs = new_action_probs.detach()

                ratio = new_action_probs / (old_action_probs + 1e-8)
                print('ratio', ratio)
                print('rewards', rewards)
                surr1 = ratio * rewards
                surr2 = torch.clamp(ratio, 1 - self.ppo_clip,
                                    1 + self.ppo_clip) * rewards
                actor_loss = (-torch.min(surr1, surr2)).mean()
                actor_loss.backward(retain_graph=False)
                optimizer.step()
                if scheduler is not None: scheduler.step()
                optimizer.zero_grad()
                model_edge.train_state.accum_step += 1

            accuracy = sum([reward == 1 for reward in rewards]) / len(rewards)
            if model.training:
                model_edge.record_metric(Metrics.TRAIN_ACC_MEAN, accuracy)
                model_edge.record_metric(Metrics.TRAIN_REWARD,
                                         torch.mean(rewards).item())
                model_edge.record_metric(Metrics.TRAIN_LOSS_MEAN,
                                         actor_loss.cpu().item())
            else:
                model_edge.record_metric(Metrics.VAL_ACC_MEAN,
                                         accuracy)

        else:
            loss = FloatTensorT([0.0])

    @overrides(BaseNLPProtocol)
    def train_epoch(self,
                    model_node: ModelNode,
                    data_loader: torch.utils.data.DataLoader[
                        ReinforcementBatch],
                    val_data_loader: torch.utils.data.DataLoader[
                        ReinforcementBatch],
                    model_edge: ModelTransitionEdge,
                    log_interval=5,
                    val_interval=50,
                    debug=False,
                    ) -> ModelNode:
        """Train a single epoch"""
        model, tokenizer = model_node.model, model_node.tokenizer
        model.train()
        set_seed(0)

        optimizer = model_edge.optimizer
        scheduler = model_edge.scheduler
        best_val_acc: float = 0.0
        model_edge.start_timer()
        first_batch = None
        for i, batch in tqdm(enumerate(data_loader), total=len(data_loader)):
            if first_batch is None:
                first_batch = batch

            loss_w_grad: FloatTensorT['1']
            accuracy: float
            self.run_batch_with_update(model, first_batch, model_edge,
                                       optimizer, scheduler, )

            model_edge.train_state.step += 1

            if (i + 1) % log_interval == 0:
                model_edge.print_train_step_info()

            if (i + 1) % val_interval == 0:
                acc_val_mean = \
                    self.val_model(val_data_loader, model_node, model_edge)[
                        Metrics.VAL_ACC_MEAN]
                if acc_val_mean > best_val_acc:
                    model_edge.store_with_identifier("BEST_VAL",
                                                     {"val_acc": acc_val_mean})
                best_val_acc = max(best_val_acc, acc_val_mean)
            settings.empty_cache()
        return model_edge.complete_model_transition()

    @overrides(BaseProtocol)
    def predict(self, model_node: ModelNode):
        pass
