import torch.nn as nn
import transformers as hug

import settings
from mgz.typing import *


# import altair as alt
# import GPUtil

class TransformerContext:
    def __init__(self, b: int, embed_len: int, n_layers: int):
        self.in_generation = False
        self.in_train = True
        self.in_test = False

        self.curr_layer = -1

        self.all_past_keys: List[FloatTensorT['B,OutSeqStep,EmbedLen']] = []
        self.all_past_values: List[FloatTensorT['B,OutSeqStep,EmbedLen']] = []
        for _ in range(n_layers):
            self.all_past_keys.append(torch.ones((b, 0, embed_len)).to(
                settings.DEVICE))
            self.all_past_values.append(torch.ones((b, 0, embed_len)).to(
                settings.DEVICE))
        # self.past_input_embeds: FloatTensorT[
        #     'B,OutSeqStep,EmbedLen'] = torch.ones((b, 0, embed_len))

    def add_key(self,
                new_key: FloatTensorT['B,OutSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_keys[self.curr_layer] = torch.cat(
            [self.all_past_keys[self.curr_layer], new_key], dim=1)
        return self.all_past_keys[self.curr_layer]

    def add_value(self,
                  new_val: FloatTensorT['B,OutSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        self.all_past_values[self.curr_layer] = torch.cat(
            [self.all_past_values[self.curr_layer], new_val], dim=1)
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


class BeamSearchContext:
    def __init__(self, n_beams: int):
        self.n_beams: int = n_beams
        self.best_probs: List[FloatTensorT['NBeams,B']] = []
        self.pred_tokens: List[LongTensorT['NBeams,B']] = []
        self.beam_indices: List[LongTensorT['NBeams,B']] = []

    def select_ids_from_probs(self,
                              probs: FloatTensorT['NBeams*B,1,VocabSize']) -> \
            LongTensorT['NBeams*B,1']:
        probs = probs.squeeze(1)
        vocab_size = probs.shape[-1]
        probs_reshape: FloatTensorT['B,NBeams,VocabSize'] = probs.view(
            self.n_beams, -1, vocab_size).permute(1, 0, 2)
        # if first step then we want to avoid resampling the same tokens from different beams, so set to neg inf
        if len(self.best_probs) == 0:
            probs_reshape[:, 1:, :] = -1e9
        probs_reshape = probs_reshape.view(-1,
                                           self.n_beams * vocab_size)
        best_probs, best_indices = \
            torch.topk(probs_reshape, k=self.n_beams, dim=-1,
                       sorted=True)

        best_probs: FloatTensorT['NBeams,B'] = \
            best_probs.permute(1, 0)
        self.best_probs.append(best_probs)

        best_indices: LongTensorT['NBeams,B,1'] = \
            best_indices.permute(1, 0)
        vocab_idx = best_indices % vocab_size
        beam_idx = best_indices // vocab_size

        print('beam_idx', vocab_idx)
        print('beam scores', best_probs.shape)
        print('beam scores', best_probs)

        self.pred_tokens.append(vocab_idx)
        self.beam_indices.append(beam_idx)

        # view 1 at the end so that it can be concatenated into sequence
        return vocab_idx.view(-1, 1)

    def get_best_sequence(self):
        seq_len = len(self.pred_tokens)
        b = self.pred_tokens[0].shape[1]
        total_probs = torch.zeros(self.n_beams, b).to(self.best_probs[0].device)
        sequence_per_beam = torch.zeros(self.n_beams, b, seq_len).to(
            self.best_probs[0].device)
        total_probs += self.best_probs[-1]
        sequence_per_beam[:, :, -1] = self.pred_tokens[-1]
        for i in range(len(self.pred_tokens) - 2, -1, -1):
            total_probs += torch.gather(self.best_probs[i], 0,
                                        self.beam_indices[i + 1])
            sequence_per_beam[:, :, i] = torch.gather(
                self.pred_tokens[i], 0,
                self.beam_indices[i + 1])
            self.beam_indices[i] = torch.gather(self.beam_indices[i], 0,
                                                self.beam_indices[i + 1])
        return sequence_per_beam[total_probs.argmax(dim=0).item(), :, :]


class LogitsRuleEnforce:
    def __init__(self, max_length: int, eos_token_id: int,
                 no_repeat_ngram_size: int = 2):
        self.eos_token_id = eos_token_id
        self.max_length = max_length
        self.no_repeat_ngram_size = no_repeat_ngram_size

    def __call__(self, beam_ctx: BeamSearchContext,
                 new_logits: FloatTensorT['n_beams*B,1,VocabSize']):
        new_logits: FloatTensorT['n_beams*B,VocabSize'] = new_logits.squeeze(1)
        new_logits[:, self.eos_token_id] = -float("inf")
        if len(beam_ctx.pred_tokens) > 0:
            prev_tokens: LongTensorT['NBeams,B'] = beam_ctx.pred_tokens[-1]
            new_logits.scatter_(-1, prev_tokens, -float("inf"))
        if len(beam_ctx.pred_tokens) == self.max_length - 1:
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

    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,OutSeqLen'],
                 src_mask: IntTensorT['B,1|OutSeqLen,SrcSeqLen']):
        n_beams = self.config.num_beams
        context = TransformerContext(src_ids.shape[0] * n_beams,
                                     self.config.d_model,
                                     self.config.decoder_layers)
        context.in_generation = True
        memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = self.encode(
            src_ids=src_ids, src_mask=src_mask)

        # prepare for beam searchza
        beam_search = BeamSearchContext(n_beams=self.config.num_beams)
        logits_rule = LogitsRuleEnforce(max_length=self.config.max_length,
                                        eos_token_id=self.config.eos_token_id)
        memory: FloatTensorT['n_beams*B,SrcSeqLen,EmbedLen'] = memory.repeat(
            n_beams, 1, 1)
        new_ids: LongTensorT['n_beams*B,1'] = tgt_ids.repeat(n_beams,
                                                             1)
        for i in range(0, 36):
            logits: FloatTensorT['n_beams*B,1,VocabSize'] = \
                self.decode(encoder_memory=memory,
                            tgt_ids=new_ids,
                            src_mask=src_mask,
                            transformer_ctx=context)
            logits = logits_rule.__call__(beam_ctx=beam_search,
                                          new_logits=logits)
            probs = torch.log_softmax(logits, dim=-1)
            new_ids = beam_search.select_ids_from_probs(probs=probs)
        return beam_search.get_best_sequence()
