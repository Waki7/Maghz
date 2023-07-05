import torch

import torch.nn as nn
import transformers as hug

import settings
from mgz.typing import *

import heapq


# import altair as alt
# import GPUtil


class BeamInference:
    def __init__(self, n_beams: int, eos_token_id: int, max_seq_len: int):
        self.n_beams: int = n_beams
        self._kill_val = -1e9
        self.best_probs: List[FloatTensorT['B,NBeams']] = []
        self.pred_tokens: List[LongTensorT['B,NBeams']] = []
        self.beam_indices: List[LongTensorT['B,NBeams']] = []

        self.eos_token_id: int = eos_token_id
        self.max_seq_len: int = max_seq_len

        # these will be initialized for every new inference
        self.done_beams: List[List[int]] = []  # batch x beam size
        self.beam_end: List[List[int]] = []  # batch x beam size
        self.lowest_done_beam_score: FloatTensorT['NBeams'] = \
            torch.ones(self.n_beams).to(settings.DEVICE) * torch.nan

        self.is_done = False
        self.length_penalty = 1

    def _all_beams_finished(self):
        return all(
            [len(beams) == self.n_beams for beams in self.done_beams])

    def select_ids_from_logprobs(self, log_probs: FloatTensorT[
        'NBeams*B,VocabSize']) -> LongTensorT['NBeams*B,1']:
        vocab_size = log_probs.shape[-1]
        batch_size = (int)(log_probs.size(0) // self.n_beams)
        seq_len = len(self.best_probs)
        if len(self.done_beams) == 0:
            [self.done_beams.append([]) for _ in range(0, batch_size)]
            [self.beam_end.append([]) for _ in range(0, batch_size)]
            self.lowest_done_beam_score = torch.ones(batch_size).to(
                settings.DEVICE) * torch.nan

        # since it's a probability conditional on the previous ones, and we
        # took the log prob, we can just add the previous probabilities
        if seq_len > 0:
            log_probs += self.best_probs[-1].view(
                batch_size * self.n_beams).unsqueeze(-1).expand_as(log_probs)

        log_probs: FloatTensorT['B,NBeams,VocabSize'] = \
            log_probs.view(batch_size, self.n_beams, vocab_size)

        # if first step then we want to avoid resampling the same tokens from different beams, so set to neg inf
        if seq_len == 0:
            log_probs[:, 1:, :] = self._kill_val
        #
        # # set beams that already predicted an eos token to -1e9
        # for batch_i, completed_beam_batch in enumerate(
        #         self.done_beams):
        #     for completed_beam in completed_beam_batch:
        #         log_probs[batch_i, completed_beam, :] = self._kill_val
        # log_probs[
        #     batch_i, completed_beam, self.eos_token_id] = -1 * self._kill_val

        log_probs = log_probs.reshape(-1,
                                      self.n_beams * vocab_size)
        topk_probs, topk_indices = \
            torch.topk(log_probs, k=2 * self.n_beams, dim=-1, largest=True,
                       sorted=True)

        topk_probs: FloatTensorT['B,NBeams'] = topk_probs
        topk_indices: LongTensorT['B,NBeams'] = topk_indices
        vocab_idx = topk_indices % vocab_size
        beam_idx = topk_indices // vocab_size

        # add beams that are done (predict eos) basically pruning the beam
        done_indices: LongTensorT['N,NDim'] = torch.argwhere(
            vocab_idx == self.eos_token_id)
        for batch, new_i in done_indices:
            from_beam = beam_idx[batch][new_i]
            topk_probs[batch][new_i] = self._kill_val
            if len(self.done_beams[batch]) < self.n_beams:
                self.done_beams[batch].append(from_beam)
                # since haven't added to best_probs yet, todo so not sure on -1
                self.beam_end[batch].append(seq_len - 1)
                self.lowest_done_beam_score[batch] = min(
                    self.lowest_done_beam_score[batch],
                    topk_probs[batch, from_beam] / (
                            seq_len ** self.length_penalty))
        # decide if done
        self.is_done = self._all_beams_finished() or (
                torch.max(topk_probs, dim=1)[
                    0] < self.lowest_done_beam_score).all() or \
                       seq_len == self.max_seq_len

        # we took extra beams earlier in case some would complete, we only want to keep exploring incomplete beams
        _, beams_to_continue = \
            torch.topk(topk_probs, k=self.n_beams, dim=1, largest=True,
                       sorted=True)

        def mps_gather_bug_workaround(_input, _dim, _index):
            # return torch.gather(_input.unsqueeze(-1), _dim,
            #                     _index.unsqueeze(-1)).squeeze(-1)
            return torch.gather(_input, _dim, _index)

        topk_probs = mps_gather_bug_workaround(topk_probs, -1,
                                               beams_to_continue)
        vocab_idx = mps_gather_bug_workaround(vocab_idx, -1, beams_to_continue)
        beam_idx = mps_gather_bug_workaround(beam_idx, -1, beams_to_continue)
        self.best_probs.append(topk_probs)
        self.pred_tokens.append(vocab_idx)
        self.beam_indices.append(beam_idx)
        # view 1 at the end so that it can be concatenated into sequence
        return vocab_idx.flatten()

    # TODO: is this finding the max prob redundant? because of log probs summing
    def get_best_sequence(self) -> LongTensorT['OutSeqLen']:
        def backtrack_beam(batch_idx, beam_num, beam_end):
            tokens: List[int] = []
            next_beam_num = beam_num
            for i in range(beam_end, -1, -1):
                tokens.insert(0,
                              self.pred_tokens[i][
                                  batch_idx, next_beam_num].item())
                next_beam_num = self.beam_indices[i][
                    batch_idx, next_beam_num].item()
            return torch.LongTensor(tokens)

        sequence_per_batch: List[LongTensorT['OutSeqLen']] = []

        for batch_idx, batch_beam_num in enumerate(self.done_beams):
            batch_beam_end = self.beam_end[batch_idx]
            best_score = float("-inf")
            best_sequence_for_batch = None

            for beam_num, beam_end in zip(batch_beam_num, batch_beam_end):
                score = self.best_probs[beam_end][batch_idx, beam_num] / (
                        (beam_end + 1) ** self.length_penalty)
                if score > best_score:
                    best_sequence_for_batch = backtrack_beam(batch_idx,
                                                             beam_num, beam_end)
                    best_score = score
            sequence_per_batch.append(best_sequence_for_batch)
        return sequence_per_batch


class TransformerContext:
    def __init__(self, b: int, embed_len: int, n_layers: int, n_heads: int):
        self.in_generation = False
        self.in_train = True
        self.in_test = False

        self.curr_layer = -1

        self.encoder_key: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.encoder_value: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.all_past_keys: List[FloatTensorT['B,OutSeqStep,EmbedLen']] = []
        self.all_past_values: List[FloatTensorT['B,OutSeqStep,EmbedLen']] = []
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
                new_key: FloatTensorT['B,OutSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_keys[self.curr_layer] = torch.cat(
            [self.all_past_keys[self.curr_layer], new_key], dim=-2)
        return self.all_past_keys[self.curr_layer]

    def add_value(self,
                  new_val: FloatTensorT['B,OutSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_values[self.curr_layer] = torch.cat(
            [self.all_past_values[self.curr_layer], new_val], dim=-2)
        return self.all_past_values[self.curr_layer]

    def get_seq_len_so_far(self):
        return self.all_past_values[0].shape[-2]

    def for_layer(self, layer: int):
        self.curr_layer = layer
        return self

    def reorder(self, beam_searcher: BeamInference):
        for i in range(0, len(self.all_past_keys)):
            # B = NBeams * B in this case
            beam_indices: LongTensorT['NBeams,B'] = \
                beam_searcher.beam_indices[-1]

            self.all_past_keys[i] = \
                self.all_past_keys[i].index_select(0,
                                                   beam_indices.flatten())
            self.all_past_values[i] = \
                self.all_past_values[i].index_select(0,
                                                     beam_indices.flatten())

    # def add_input_embeddings(self,
    #                          new_val: FloatTensorT['B,OutSeqStep,EmbedLen']):
    #     self.past_input_embeds = torch.cat([self.past_input_embeds, new_val],
    #                                        dim=1)
    #     return self.past_input_embeds

    def reset(self):
        del self


class LogitsRuleEnforce:
    def __init__(self, max_length: int, eos_token_id: int,
                 no_repeat_ngram_size: int = 2):
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, input_ids: LongTensorT['B,OutSeqLen'],
                 new_logits: FloatTensorT['n_beams*B,VocabSize'], seq_len: int):
        # new_logits[:, self.eos_token_id] = -float("inf")
        if seq_len > 0:
            prev_tokens: LongTensorT['NBeams,B'] = input_ids
            new_logits.scatter_(-1, prev_tokens, -float("inf"))
        if seq_len == (self.max_length - 1):
            new_logits[:, self.eos_token_id] = 1e9
        return new_logits


class BaseTransformer(nn.Module):
    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        self.config: hug.PretrainedConfig = config

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def decode(self,
               encoder_memory: FloatTensorT['B,SrcSeqLen,EmbedLen'],
               tgt_ids: LongTensorT['B,OutSeqLen'],
               src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen'],
               transformer_ctx: TransformerContext,
               tgt_mask: IntTensorT['B,OutSeqLen,OutSeqLen'] = None):
        raise NotImplementedError

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,OutSeqLen'],
            src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,OutSeqLen,OutSeqLen']
    ) -> FloatTensorT['B,OutSeqLen,EmbedLen']:
        raise NotImplementedError

    #
    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,OutSeqLen'] = None
                 ) -> \
            LongTensorT['OutSeqLen']:
        if tgt_ids is None:
            tgt_ids = torch.LongTensor([config.sep_token_id]).unsqueeze(0).to(
                settings.DEVICE).repeat(src_ids.shape[0], 1)
        n_beams = self.config.num_beams
        max_len = self.config.max_length
        eos_token_id = self.config.eos_token_id
        context = TransformerContext(src_ids.shape[0] * n_beams,
                                     self.config.d_model,
                                     self.config.decoder_layers,
                                     self.config.num_attention_heads)
        context.in_generation = True
        memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = self.encode(
            src_ids=src_ids, src_mask=src_mask)

        # prepare for beam searcher
        beam_search = BeamInference(n_beams=n_beams,
                                    eos_token_id=eos_token_id,
                                    max_seq_len=max_len)
        logits_rule = LogitsRuleEnforce(max_length=max_len,
                                        eos_token_id=eos_token_id)
        memory: FloatTensorT['n_beams*B,SrcSeqLen,EmbedLen'] = memory.repeat(
            n_beams, 1, 1)
        new_ids: LongTensorT['n_beams*B,1'] = tgt_ids.repeat(n_beams,
                                                             1)
        src_mask = src_mask.repeat(n_beams, 1)

        for i in range(0, max_len):
            logits: FloatTensorT['n_beams*B,VocabSize'] = \
                self.decode(encoder_memory=memory,
                            tgt_ids=new_ids,
                            src_mask=src_mask,
                            transformer_ctx=context).squeeze(-2)
            logits = logits_rule.__call__(input_ids=new_ids,
                                          new_logits=logits,
                                          seq_len=i)

            probs = torch.log_softmax(logits, dim=-1)
            new_ids: LongTensorT[
                'n_beams*B,1'] = beam_search.select_ids_from_logprobs(
                log_probs=probs).unsqueeze(-1)
            # todo let's make this a function of beam search instead, isn't obvious with the current pattern
            context.reorder(beam_search)
            if beam_search.is_done:
                break
        return beam_search.get_best_sequence()
