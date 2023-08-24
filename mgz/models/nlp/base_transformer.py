import torch.nn as nn
import torch.nn as nn
import transformers as hug
from transformers import PreTrainedTokenizerBase
from transformers.generation.utils import MinLengthLogitsProcessor, \
    ForcedEOSTokenLogitsProcessor, \
    NoRepeatNGramLogitsProcessor, LogitsProcessorList, MaxLengthCriteria, \
    BeamSearchScorer

import settings
from mgz.typing import *


# import altair as alt
# import GPUtil


class TransformerContext:
    def __init__(self, b: int, embed_len: int, n_layers: int, n_heads: int):
        self.in_generation = False
        self.in_train = True
        self.in_test = False

        self.curr_layer = -1

        self.encoder_key: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.encoder_value: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.all_past_keys: List[FloatTensorT['B,TgtSeqStep,EmbedLen']] = []
        self.all_past_values: List[FloatTensorT['B,TgtSeqStep,EmbedLen']] = []
        for _ in range(n_layers):
            self.all_past_keys.append(
                torch.ones((b, n_heads, 0, embed_len // n_heads)).to(
                    settings.DEVICE))
            self.all_past_values.append(
                torch.ones((b, n_heads, 0, embed_len // n_heads)).to(
                    settings.DEVICE))
        # self.past_input_embeds: FloatTensorT[
        #     'B,OutSeqStep,EmbedLen'] = torch.ones((b, 0, embed_len))

    def add_key(self,
                new_key: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_keys[self.curr_layer] = torch.cat(
            [self.all_past_keys[self.curr_layer], new_key], dim=-2)
        return self.all_past_keys[self.curr_layer]

    def add_value(self,
                  new_val: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_values[self.curr_layer] = torch.cat(
            [self.all_past_values[self.curr_layer], new_val], dim=-2)
        return self.all_past_values[self.curr_layer]

    def get_seq_len_so_far(self):
        return self.all_past_values[0].shape[-2]

    def for_layer(self, layer: int):
        self.curr_layer = layer
        return self

    # def add_input_embeddings(self,
    #                          new_val: FloatTensorT['B,OutSeqStep,EmbedLen']):
    #     self.past_input_embeds = torch.cat([self.past_input_embeds, new_val],
    #                                        dim=1)
    #     return self.past_input_embeds

    def reset(self):
        del self


class Greedy:
    def __init__(self, config: hug.PretrainedConfig, batch_size: int):
        self.n_beams: int = config.num_beams
        self.eos_token_id: int = config.eos_token_id
        self.pad_token_id: int = config.pad_token_id
        self.max_seq_len: int = config.max_length
        # these will be initialized for every new inference
        self.is_done = False
        self.sequence: List[FloatTensorT['B,VocabSize']] = []
        self.length_penalty = config.length_penalty
        self.logits_processors = LogitsProcessorList([
            NoRepeatNGramLogitsProcessor(config.no_repeat_ngram_size)])
        if config.min_length > 0:
            self.logits_processors.append(
                MinLengthLogitsProcessor(min_length=config.min_length,
                                         eos_token_id=config.eos_token_id))
        if config.forced_eos_token_id is not None:
            self.logits_processors.append(
                ForcedEOSTokenLogitsProcessor(max_length=config.max_length,
                                              eos_token_id=config.eos_token_id))
        self.stopping_criteria = MaxLengthCriteria(max_length=config.max_length)

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
        return torch.stack(self.sequence, dim=-1)


class BeamInference:
    def __init__(self, config: hug.PretrainedConfig, batch_size: int):

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
        self.max_seq_len: int = config.max_length

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
                ForcedEOSTokenLogitsProcessor(max_length=config.max_length,
                                              eos_token_id=config.eos_token_id))
        self.stopping_criteria = MaxLengthCriteria(max_length=config.max_length)

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


class LogitsRuleEnforce:
    def __init__(self, max_length: int, eos_token_id: int,
                 no_repeat_ngram_size: int = 2):
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, input_ids: LongTensorT['B,TgtSeqLen'],
                 new_logits: FloatTensorT['n_beams*B,VocabSize'], seq_len: int):
        # new_logits[:, self.eos_token_id] = -float("inf")
        if seq_len > 0:
            prev_tokens: LongTensorT['NBeams,B'] = input_ids
            new_logits.scatter_(-1, prev_tokens, -float("inf"))
        if seq_len == (self.max_length - 1):
            new_logits[:, self.eos_token_id] = 1e4
        return new_logits


class BaseTransformer(nn.Module):
    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        self.config: hug.PretrainedConfig = config

    def verify_tokenizer(self, tokenizer: PreTrainedTokenizerBase):
        def validate_field(field_name):
            assert hasattr(tokenizer, field_name), "tokenizer missing {} field"
            field_value = getattr(tokenizer, field_name)
            if not hasattr(self.config, field_name) or getattr(self.config,
                                                               field_name) is None:
                setattr(self.config, field_name, field_value)
            else:
                assert getattr(self.config, field_name) == field_value, \
                    "config {} vs tokenizer {}".format(
                        getattr(self.config, field_name), field_value)

        validate_field('pad_token_id')
        validate_field('bos_token_id')
        validate_field('eos_token_id')
        validate_field('sep_token_id')
        validate_field('cls_token_id')
        validate_field('mask_token_id')
        validate_field('vocab_size')

    # For LED we need to pad the input ids and attention mask to be multiple of the attention window
    def _pre_encode_pad_if_needed(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,SrcSeqLen'],
            pad_token_id: int,
    ):
        return src_ids, src_mask

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def decode(self,
               encoder_memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
               tgt_ids: LongTensorT['B,TgtSeqLen'],
               src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
               transformer_ctx: TransformerContext,
               tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'] = None):
        raise NotImplementedError

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen']
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        raise NotImplementedError

    #
    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,TgtSeqLen'] = None
                 ) -> \
            List[LongTensorT['TgtSeqLen']]:
        if tgt_ids is None:
            # todo@ceyer don't like using config for sep token id
            tgt_ids = torch.LongTensor([self.config.sep_token_id]).unsqueeze(
                0).to(settings.DEVICE).repeat(src_ids.shape[0], 1)

        n_beams = self.config.num_beams
        max_len = self.config.max_length
        context = TransformerContext(src_ids.shape[0] * n_beams,
                                     self.config.d_model,
                                     self.config.decoder_layers,
                                     self.config.num_attention_heads)
        context.in_generation = True
        memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = self.encode(
            src_ids=src_ids, src_mask=src_mask)

        # prepare for beam searcher
        if n_beams > 1:
            beam_search = BeamInference(config=self.config,
                                        batch_size=src_ids.shape[0])
        else:
            beam_search = Greedy(config=self.config,
                                 batch_size=src_ids.shape[0])

        memory: FloatTensorT['n_beams*B,SrcSeqLen,EmbedLen'] = memory.repeat(
            n_beams, 1, 1)
        new_ids: LongTensorT['n_beams*B,1'] = tgt_ids.repeat(n_beams,
                                                             1)
        input_ids: LongTensorT['n_beams*B,TgtSeqStep'] = new_ids.clone()

        src_mask = src_mask.repeat(n_beams, 1)
        for i in range(0, max_len):
            logits: FloatTensorT['n_beams*B,VocabSize'] = \
                self.decode(encoder_memory=memory,
                            tgt_ids=new_ids,
                            src_mask=src_mask,
                            transformer_ctx=context).squeeze(-2)
            probs = torch.log_softmax(logits, dim=-1)
            new_ids: LongTensorT[
                'n_beams*B, 1'] = beam_search.select_ids_from_logprobs(
                log_probs=probs, input_ids=input_ids).unsqueeze(-1)
            beam_search.reoder_transformer_context(context)
            input_ids = torch.cat([input_ids, new_ids], dim=-1)
            if beam_search.is_done:
                break
        seq_output: LongTensorT['B,TgtSeqLen'] = \
            beam_search.get_best_sequence(input_ids)
        return seq_output

# class BeamInference2:
#     def __init__(self, n_beams: int, eos_token_id: int, max_seq_len: int,
#                  length_penalty=1):
#         self.n_beams: int = n_beams
#         self._kill_val = -1e9
#         self.best_probs: List[FloatTensorT['B,NBeams']] = []
#         self.pred_tokens: List[LongTensorT['B,NBeams']] = []
#         self.beam_indices: List[LongTensorT['B,NBeams']] = []
#
#         self.eos_token_id: int = eos_token_id
#         self.max_seq_len: int = max_seq_len
#
#         # these will be initialized for every new inference
#         self.done_beams: List[List[int]] = []  # batch x beam size
#         self.beam_end: List[List[int]] = []  # batch x beam size
#         self.lowest_done_beam_score: FloatTensorT['NBeams'] = \
#             torch.ones(self.n_beams).to(settings.DEVICE) * torch.nan
#
#         self.is_done = False
#         self.length_penalty = length_penalty
#
#     def _all_beams_finished(self):
#         return all(
#             [len(beams) == self.n_beams for beams in self.done_beams])
#
#     def select_ids_from_logprobs(self, log_probs: FloatTensorT[
#         'NBeams*B,VocabSize']) -> LongTensorT['NBeams*B,1']:
#         vocab_size = log_probs.shape[-1]
#         batch_size = (int)(log_probs.size(0) // self.n_beams)
#         seq_len = len(self.best_probs)
#         if len(self.done_beams) == 0:
#             [self.done_beams.append([]) for _ in range(0, batch_size)]
#             [self.beam_end.append([]) for _ in range(0, batch_size)]
#             self.lowest_done_beam_score = torch.ones(batch_size).to(
#                 settings.DEVICE) * torch.nan
#
#         # since it's a probability conditional on the previous ones, and we
#         # took the log prob, we can just add the previous probabilities
#         if seq_len > 0:
#             log_probs += self.best_probs[-1].view(
#                 batch_size * self.n_beams).unsqueeze(-1).expand_as(log_probs)
#
#         log_probs: FloatTensorT['B,NBeams,VocabSize'] = \
#             log_probs.view(batch_size, self.n_beams, vocab_size)
#
#         # if first step then we want to avoid resampling the same tokens from different beams, so set to neg inf
#         if seq_len == 0:
#             log_probs[:, 1:, :] = self._kill_val
#         #
#         # # set beams that already predicted an eos token to -1e9
#         # for batch_i, completed_beam_batch in enumerate(
#         #         self.done_beams):
#         #     for completed_beam in completed_beam_batch:
#         #         log_probs[batch_i, completed_beam, :] = self._kill_val
#         # log_probs[
#         #     batch_i, completed_beam, self.eos_token_id] = -1 * self._kill_val
#
#         log_probs = log_probs.reshape(-1,
#                                       self.n_beams * vocab_size)
#         topk_probs, topk_indices = \
#             torch.topk(log_probs, k=2 * self.n_beams, dim=-1, largest=True,
#                        sorted=True)
#
#         topk_probs: FloatTensorT['B,NBeams'] = topk_probs
#         topk_indices: LongTensorT['B,NBeams'] = topk_indices
#         vocab_idx = topk_indices % vocab_size
#         beam_idx = topk_indices // vocab_size
#
#         # add beams that are done (predict eos) basically pruning the beam
#         done_indices: LongTensorT['N,NDim'] = torch.argwhere(
#             vocab_idx == self.eos_token_id)
#         for batch, new_i in done_indices:
#             from_beam = beam_idx[batch][new_i]
#             topk_probs[batch][new_i] = self._kill_val
#             if len(self.done_beams[batch]) < self.n_beams:
#                 self.done_beams[batch].append(from_beam)
#                 # since haven't added to best_probs yet, todo so not sure on -1
#                 self.beam_end[batch].append(seq_len - 1)
#                 self.lowest_done_beam_score[batch] = min(
#                     self.lowest_done_beam_score[batch],
#                     topk_probs[batch, from_beam] / (
#                             seq_len ** self.length_penalty))
#         # decide if done
#         self.is_done = self._all_beams_finished() or (
#                 torch.max(topk_probs, dim=1)[
#                     0] < self.lowest_done_beam_score).all() or \
#                        seq_len == self.max_seq_len
#
#         # we took extra beams earlier in case some would complete, we only want to keep exploring incomplete beams
#         _, beams_to_continue = \
#             torch.topk(topk_probs, k=self.n_beams, dim=1, largest=True,
#                        sorted=True)
#
#         def mps_gather_bug_workaround(_input, _dim, _index):
#             # return torch.gather(_input.unsqueeze(-1), _dim,
#             #                     _index.unsqueeze(-1)).squeeze(-1)
#             return torch.gather(_input, _dim, _index)
#
#         topk_probs = mps_gather_bug_workaround(topk_probs, -1,
#                                                beams_to_continue)
#         vocab_idx = mps_gather_bug_workaround(vocab_idx, -1, beams_to_continue)
#         beam_idx = mps_gather_bug_workaround(beam_idx, -1, beams_to_continue)
#         self.best_probs.append(topk_probs)
#         self.pred_tokens.append(vocab_idx)
#         self.beam_indices.append(beam_idx)
#
#         # view 1 at the end so that it can be concatenated into sequence
#         return vocab_idx.flatten()
#
#     # TODO: is this finding the max prob redundant? because of log probs summing
#     def get_best_sequence(self) -> List[LongTensorT['TgtSeqLen']]:
#         def backtrack_beam(batch_idx, beam_num, beam_end):
#             tokens: List[int] = []
#             next_beam_num = beam_num
#             for i in range(beam_end, -1, -1):
#                 tokens.insert(0,
#                               self.pred_tokens[i][
#                                   batch_idx, next_beam_num].item())
#                 next_beam_num = self.beam_indices[i][
#                     batch_idx, next_beam_num].item()
#             return torch.LongTensor(tokens)
#
#         sequence_per_batch: List[LongTensorT['TgtSeqLen']] = []
#
#         for batch_idx, batch_beam_num in enumerate(self.done_beams):
#             batch_beam_end = self.beam_end[batch_idx]
#             best_score = float("-inf")
#             best_sequence_for_batch = None
#
#             for beam_num, beam_end in zip(batch_beam_num, batch_beam_end):
#                 score = self.best_probs[beam_end][batch_idx, beam_num] / (
#                         (beam_end + 1) ** self.length_penalty)
#                 if score > best_score:
#                     best_sequence_for_batch = backtrack_beam(batch_idx,
#                                                              beam_num, beam_end)
#                     best_score = score
#             sequence_per_batch.append(best_sequence_for_batch)
#         return sequence_per_batch
