from __future__ import annotations

import transformers as hug
from transformers.generation.utils import MinLengthLogitsProcessor, \
    ForcedEOSTokenLogitsProcessor, \
    NoRepeatNGramLogitsProcessor, LogitsProcessorList, MaxLengthCriteria, \
    BeamSearchScorer

import mgz.settings as settings
from mgz.typing import *

if TYPE_CHECKING:
    from mgz.models.nlp.base_transformer import TransformerContext
from mgz.typing import *


class Greedy:
    def __init__(self, config: hug.PretrainedConfig, batch_size: int,
                 max_tokens: int = None):
        self.n_beams: int = config.num_beams
        self.eos_token_id: int = config.eos_token_id
        self.pad_token_id: int = config.pad_token_id
        self.max_seq_len: int = config.max_length if max_tokens is None else \
            max_tokens
        # these will be initialized for every new inference
        self.is_done = False
        self.sequence: List[FloatTensorT['B,VocabSize']] = []
        self.length_penalty = config.length_penalty

        self.logits_processors = LogitsProcessorList([])
        if hasattr(config,
                   'no_repeat_ngram_size') and config.no_repeat_ngram_size > 0:
            self.logits_processors.append(
                NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size))
        if config.min_length > 0:
            self.logits_processors.append(
                MinLengthLogitsProcessor(min_length=config.min_length,
                                         eos_token_id=config.eos_token_id))
        if config.forced_eos_token_id is not None:
            self.logits_processors.append(
                ForcedEOSTokenLogitsProcessor(max_length=self.max_seq_len,
                                              eos_token_id=config.eos_token_id))
        self.stopping_criteria = MaxLengthCriteria(max_length=self.max_seq_len)

    def select_ids_from_logprobs(self, log_probs: FloatTensorT[
        'B,VocabSize'], input_ids: LongTensorT['N,TgtSeqStep']) -> \
            LongTensorT['NBeams*B']:
        batch_size = log_probs.shape[0]
        next_token_scores = log_probs
        next_token_scores_processed = self.logits_processors(input_ids,
                                                             next_token_scores)

        next_token_scores, next_tokens = torch.max(
            next_token_scores_processed, dim=1, keepdim=False
        )
        self.sequence.append(next_tokens)
        self.is_done = self.stopping_criteria(input_ids, next_token_scores)
        # todo, not the best, cache for call to finalizing later
        return next_tokens

    def reoder_transformer_context(self,
                                   transformer_context: TransformerContext):
        return

    def get_best_sequence(self,
                          input_ids: LongTensorT['NBeams*B,TgtSeqStep']) -> \
            LongTensorT['TgtSeqLen']:
        return input_ids


class BeamInference:
    def __init__(self, config: hug.PretrainedConfig, batch_size: int,
                 max_tokens: int = None):

        self.n_beams: int = config.num_beams
        self.inference_scorer = BeamSearchScorer(
            batch_size=batch_size,
            num_beams=self.n_beams,
            device=settings.DEVICE,
        )
        self.beam_scores = torch.zeros((batch_size, self.n_beams),
                                       dtype=torch.float,
                                       device=settings.DEVICE)
        self.beam_scores[:, 1:] = -1e9
        self.beam_scores = self.beam_scores.view(
            (batch_size * self.n_beams,))

        self.eos_token_id: int = config.eos_token_id
        self.pad_token_id: int = config.pad_token_id
        self.max_seq_len: int = config.max_length if max_tokens is None else \
            max_tokens

        # these will be initialized for every new inference
        self.is_done = False
        self.beam_idx = None  # this is used to reshuffle the transformer context
        self.next_tokens = None
        self.next_indices = None
        self.length_penalty = config.length_penalty
        self.logits_processors = LogitsProcessorList([
            NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size)])
        if config.min_length > 0:
            self.logits_processors.append(
                MinLengthLogitsProcessor(min_length=config.min_length,
                                         eos_token_id=config.eos_token_id))
        if config.forced_eos_token_id is not None:
            self.logits_processors.append(
                ForcedEOSTokenLogitsProcessor(max_length=self.max_seq_len,
                                              eos_token_id=config.eos_token_id))
        self.stopping_criteria = MaxLengthCriteria(max_length=self.max_seq_len)

    def _all_beams_finished(self):
        return all(
            [len(beams) == self.n_beams for beams in self.done_beams])

    def select_ids_from_logprobs(self, log_probs: FloatTensorT[
        'NBeams*B,VocabSize'], input_ids: LongTensorT['NBeams*B,TgtSeqStep']) -> \
            LongTensorT['NBeams*B']:
        batch_size = log_probs.shape[0] // self.n_beams

        next_token_scores = log_probs
        next_token_scores_processed = self.logits_processors(input_ids,
                                                             next_token_scores)
        next_token_scores = next_token_scores_processed + self.beam_scores[:,
                                                          None].expand_as(
            next_token_scores)
        vocab_size = next_token_scores.shape[-1]
        next_token_scores = next_token_scores.view(batch_size,
                                                   self.n_beams * vocab_size)
        # Sample 2 next tokens for each beam (so we have some spare tokens and match output of beam search)
        next_token_scores, next_tokens = torch.topk(
            next_token_scores, 2 * self.n_beams, dim=1, largest=True,
            sorted=True
        )
        next_indices = torch.div(next_tokens, vocab_size, rounding_mode="floor")
        next_tokens = next_tokens % vocab_size
        beam_outputs = self.inference_scorer.process(
            input_ids,
            next_token_scores,
            next_tokens,
            next_indices,
            pad_token_id=self.pad_token_id,
            eos_token_id=self.eos_token_id,
        )
        self.beam_scores = beam_outputs["next_beam_scores"]
        beam_next_tokens = beam_outputs["next_beam_tokens"]
        self.beam_idx = beam_outputs["next_beam_indices"]
        self.is_done = self.inference_scorer.is_done or self.stopping_criteria(
            input_ids, next_token_scores)
        # todo, not the best, cache for call to finalizing later
        if self.is_done:
            self.next_tokens = next_tokens
            self.next_indices = next_indices
        return beam_next_tokens

    def reoder_transformer_context(self,
                                   transformer_context: TransformerContext):
        for i in range(0, len(transformer_context.all_past_keys)):
            # B = NBeams * B in this case
            beam_indices: LongTensorT['NBeams,B'] = \
                self.beam_idx

            transformer_context.all_past_keys[i] = \
                transformer_context.all_past_keys[i].index_select(0,
                                                                  beam_indices.flatten())
            transformer_context.all_past_values[i] = \
                transformer_context.all_past_values[i].index_select(0,
                                                                    beam_indices.flatten())

    def get_best_sequence(self,
                          input_ids: LongTensorT['NBeams*B,TgtSeqStep']) -> \
            LongTensorT['TgtSeqLen']:
        return self.inference_scorer.finalize(input_ids=input_ids,
                                              final_beam_scores=self.beam_scores,
                                              final_beam_tokens=self.next_tokens,
                                              final_beam_indices=self.next_indices,
                                              pad_token_id=self.pad_token_id,
                                              eos_token_id=self.eos_token_id,
                                              max_length=self.max_seq_len)[
            'sequences']
