from __future__ import annotations

import bitsandbytes
import torch.nn as nn
import transformers as hug
from accelerate import dispatch_model
from bitsandbytes.nn import Linear8bitLt
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from transformers import PreTrainedTokenizer
from transformers.generation.utils import MinLengthLogitsProcessor, \
    ForcedEOSTokenLogitsProcessor, \
    NoRepeatNGramLogitsProcessor, LogitsProcessorList, MaxLengthCriteria, \
    BeamSearchScorer
from transformers.integrations import replace_with_bnb_linear

import settings
from mgz.models.base_model import BaseModel
from mgz.typing import *


# import altair as alt
# import GPUtil


def replace_8bit_linear(model, threshold=6.0, module_to_not_convert=None):
    if module_to_not_convert is None:
        module_to_not_convert = ["lm_head"]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_not_convert)

        if isinstance(module, nn.Linear) and name not in module_to_not_convert:
            # with init_empty_weights():
            model._modules[name] = Linear8bitLt(
                module.in_features,
                module.out_features,
                module.bias is not None,
                has_fp16_weights=False,
                threshold=threshold,
            )
        # if isinstance(module,
        #               nn.Embedding) and name not in module_to_not_convert:
        #     model._modules[name] = bitsandbytes.nn.StableEmbedding(
        #         num_embeddings=module.num_embeddings,
        #         embedding_dim=module.embedding_dim,
        #         padding_idx=module.padding_idx)
    return model


def quantize_model(model: BaseTransformer):
    model = replace_8bit_linear(
        model,
        module_to_not_convert=model.modules_to_not_convert()
    )
    model.config.quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    return model


def quantize_model_inference(model: BaseTransformer):
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        device_map="auto",
    )
    if quantization_config is not None:
        model = replace_with_bnb_linear(
            model,
            modules_to_not_convert=model.modules_to_not_convert(),
            quantization_config=quantization_config
        )
        model.config.quantization_config = quantization_config
    model = prepare_model_for_kbit_training(model)
    model = dispatch_model(model, device_map={"": settings.DEVICE})
    return model


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


class BaseTransformer(BaseModel):

    @classmethod
    def modules_to_apply_lora(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def modules_to_not_convert(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def load_model(cls, path: str) -> Self:
        raise NotImplementedError

    @classmethod
    def load_tokenizer(cls, tokenizer_id: str) -> hug.PreTrainedTokenizer:
        raise NotImplementedError

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        raise NotImplementedError

    @abstractmethod
    def save(self, path: DirPath):
        raise NotImplementedError

    def __init__(self, config):
        super(BaseTransformer, self).__init__()
        self.config: hug.PretrainedConfig = config

    def prepare_inputs_for_generation(self):
        pass

    def verify_tokenizer(self, tokenizer: PreTrainedTokenizer):
        def validate_field(field_name):
            assert hasattr(tokenizer, field_name), "tokenizer missing {} field"
            field_value = getattr(tokenizer, field_name)
            if not hasattr(self.config, field_name) or getattr(self.config,
                                                               field_name) is None:
                setattr(self.config, field_name, field_value)
            else:
                if field_name == 'vocab_size':
                    if not getattr(self.config,
                                   'vocab_size') == tokenizer.vocab_size:
                        logging.warning(
                            # TODO FIXME or understand why different
                            "config {} vs tokenizer vocab size {} + {} added "
                            "tokens for vocab_size".format(
                                getattr(self.config, field_name), field_value,
                                2))
                        # len(tokenizer.added_tokens_encoder)))
                else:
                    assert getattr(self.config, field_name) == field_value, \
                        "config {} vs tokenizer {} for field {}".format(
                            getattr(self.config, field_name), field_value,
                            field_name)

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
                 ) -> LongTensorT['TgtSeqLen']:
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


class BinaryTaggerMixin(BaseModel):
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen']
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.led.forward(
            src_ids=src_ids, src_mask=src_mask,
            tgt_ids=tgt_ids, tgt_mask=tgt_mask)

        lm_logits = self.lm_head(output)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits
