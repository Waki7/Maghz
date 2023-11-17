from __future__ import annotations

import torch.nn as nn
import transformers as hug
from accelerate import dispatch_model, init_empty_weights
from bitsandbytes.nn import Linear8bitLt
from peft import prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
from transformers import PreTrainedTokenizer
from transformers.integrations import replace_with_bnb_linear

import mgz.settings as settings
from mgz.models.base_model import BaseModel
from mgz.models.nlp.inference import BeamInference, Greedy
from mgz.typing import *


def replace_8bit_linear(model, threshold=6.0, module_to_not_convert=None):
    if module_to_not_convert is None:
        module_to_not_convert = ["lm_head"]
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            replace_8bit_linear(module, threshold, module_to_not_convert)

        if isinstance(module, nn.Linear) and name not in module_to_not_convert:
            with init_empty_weights():
                model._modules[name] = Linear8bitLt(
                    module.in_features,
                    module.out_features,
                    module.bias is not None,
                    has_fp16_weights=False,
                    threshold=threshold,
                )
    return model


def quantize_model(model: BaseTransformer):
    logging.info('Quantizing model.')
    model = replace_8bit_linear(
        model,
        module_to_not_convert=model.modules_to_not_convert()
    )
    model.config.quantization_config = BitsAndBytesConfig(
        load_in_8bit=True,
    )
    return model


def quantize_model_inference(model: BaseTransformer):
    logging.info('Quantizing model.')
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
    def __init__(self, b: int, embed_len: int, n_layers: int, n_heads: int,
                 n_key_value_heads: int = None,
                 encoder_memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None,
                 in_generation: bool = True):
        """
        n_key_value_heads can be less than n_heads. This occurs when a model
        just uses kv heads and repeats them to limit computation.
        """
        if n_key_value_heads is None:
            n_key_value_heads = n_heads
        if encoder_memory is not None:
            self.encoder_memory = encoder_memory

        self.in_generation = in_generation
        self.in_train = True
        self.in_test = False

        self.curr_layer = -1

        self.encoder_key: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.encoder_value: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
        self.all_past_keys: List[FloatTensorT['B,TgtSeqStep,EmbedLen']] = []
        self.all_past_values: List[FloatTensorT['B,TgtSeqStep,EmbedLen']] = []
        for _ in range(n_layers):
            self.all_past_keys.append(
                torch.ones((b, n_key_value_heads, 0, embed_len // n_heads)).to(
                    settings.DEVICE))
            self.all_past_values.append(
                torch.ones((b, n_key_value_heads, 0, embed_len // n_heads)).to(
                    settings.DEVICE))

    def add_key(self,
                new_key: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'

        self.all_past_keys[self.curr_layer] = torch.cat(
            [self.all_past_keys[self.curr_layer], new_key], dim=-2).to(
            new_key.dtype)
        return self.all_past_keys[self.curr_layer]

    def add_value(self,
                  new_val: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'

        self.all_past_values[self.curr_layer] = torch.cat(
            [self.all_past_values[self.curr_layer], new_val], dim=-2).to(
            new_val.dtype)
        return self.all_past_values[self.curr_layer]

    def set_key(self,
                key: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        if key.shape[1] > 1:
            self.all_past_keys[self.curr_layer] = key
            return self.all_past_keys[self.curr_layer]

    def set_value(self,
                  val: FloatTensorT['B,TgtSeqStep,EmbedLen']):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        if val.shape[1] > 1:
            self.all_past_values[self.curr_layer] = val
            return self.all_past_values[self.curr_layer]

    def get_key(self):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        return self.all_past_keys[self.curr_layer]

    def get_value(self):
        assert self.curr_layer > -1, 'call context for a specific layer number'
        return self.all_past_values[self.curr_layer]

    def get_seq_len_so_far(self):
        return self.all_past_values[self.curr_layer].shape[-2]

    def for_layer(self, layer: int):
        self.curr_layer = layer
        return self

    def reset(self):
        del self


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
    def is_encoder_decoder(cls) -> bool:
        raise NotImplementedError

    @classmethod
    def modules_to_not_convert(cls) -> List[str]:
        raise NotImplementedError

    @classmethod
    def load_model(cls, path: str,
                   quantization_config: BitsAndBytesConfig = None) -> Self:
        raise NotImplementedError

    @classmethod
    def load_tokenizer(cls, path: DirPath) -> hug.PreTrainedTokenizer:
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

        self.embed_dim: int = None
        if hasattr(config, 'd_model'):
            self.embed_dim = config.d_model
        if hasattr(config, 'hidden_size'):
            self.embed_dim = config.hidden_size
        if hasattr(config, 'embed_dim'):
            self.vocab_size = config.embed_dim

        self.n_decoder_layers: int = None
        if hasattr(config, 'decoder_layers'):
            self.n_decoder_layers = config.decoder_layers
        if hasattr(config, 'num_hidden_layers'):
            self.n_decoder_layers = config.num_hidden_layers

        self.n_attention_heads: int = None
        if hasattr(config, 'num_attention_heads'):
            self.n_attention_heads = config.num_attention_heads

        if hasattr(config, 'num_key_value_heads'):
            self.n_key_value_heads = config.num_key_value_heads
        else:
            self.n_key_value_heads = self.n_attention_heads

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

    def decode(self, **kwargs) -> (
            LogitsTensorT)['B,1,VocabSize']:
        raise NotImplementedError

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,TgtSeqLen'] = None,
                 max_new_tokens: int = None,
                 ) -> LongTensorT['TgtSeqLen']:
        if tgt_ids is None:
            tgt_ids = torch.LongTensor([self.config.sep_token_id]).unsqueeze(
                0).to(settings.DEVICE).repeat(src_ids.shape[0], 1)
        if max_new_tokens is None:
            max_new_tokens = self.config.max_length
        max_tokens = max_new_tokens + tgt_ids.shape[1]

        n_beams = self.config.num_beams
        max_len = self.config.max_length

        if self.is_encoder_decoder():
            memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = self.encode(
                src_ids=src_ids, src_mask=src_mask)
            memory: FloatTensorT[
                'n_beams*B,SrcSeqLen,EmbedLen'] = memory.repeat(
                n_beams, 1, 1)
            src_mask = src_mask.repeat(n_beams, 1)
        else:
            memory: FloatTensorT['B,SrcSeqLen,EmbedLen'] = None
            src_mask = None

        context = TransformerContext(src_ids.shape[0] * n_beams,
                                     self.embed_dim,
                                     self.n_decoder_layers,
                                     n_heads=self.n_attention_heads,
                                     n_key_value_heads=self.n_key_value_heads,
                                     encoder_memory=memory)

        # prepare for beam searcher
        if n_beams > 1:
            beam_search = BeamInference(config=self.config,
                                        batch_size=src_ids.shape[0],
                                        max_tokens=max_tokens)
        else:
            beam_search = Greedy(config=self.config,
                                 batch_size=src_ids.shape[0],
                                 max_tokens=max_tokens)

        new_ids: LongTensorT['n_beams*B,1'] = tgt_ids.repeat(n_beams, 1)
        all_ids: LongTensorT['n_beams*B,TgtSeqStep'] = new_ids.clone()

        for i in range(0, max_len):
            logits: FloatTensorT['n_beams*B,VocabSize'] = \
                self.decode(generation_ids=new_ids,
                            src_mask=src_mask,
                            transformer_ctx=context)[:, -1, :]
            probs = FloatTensorT(torch.log_softmax(logits, dim=-1))
            new_ids: LongTensorT[
                'n_beams*B,1'] = beam_search.select_ids_from_logprobs(
                log_probs=probs, input_ids=all_ids).unsqueeze(-1)
            beam_search.reoder_transformer_context(context)
            all_ids = LongTensorT(torch.cat([all_ids, new_ids], dim=-1))
            if beam_search.is_done:
                break
        seq_output: LongTensorT['B,TgtSeqLen'] = \
            beam_search.get_best_sequence(all_ids)
        return seq_output


class EncoderDecoderTransformer(BaseTransformer):

    @classmethod
    def is_encoder_decoder(cls) -> bool:
        return True

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen']
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        raise NotImplementedError

    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1'],
               src_mask: IntTensorT['B,SrcSeqLen'] = None) -> (
            LogitsTensorT)['B,1,VocabSize']:
        raise NotImplementedError

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError

    def encoder_decoder_embedding(self,
                                  src_ids: LongTensorT['B,SrcSeqLen'],
                                  tgt_ids: LongTensorT['B,TgtSeqLen'],
                                  src_mask: IntTensorT['B,SrcSeqLen'],
                                  tgt_mask: IntTensorT['B,TgtSeqLen']):
        raise NotImplementedError


class DecoderTransformer(BaseTransformer):

    @classmethod
    def is_encoder_decoder(cls) -> bool:
        return False

    def forward(
            self,
            src_ids: LongTensorT['B,TgtSeqLen'],
            transformer_ctx: TransformerContext = None,
            src_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'] = None
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        raise NotImplementedError

    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1']) -> (
            LogitsTensorT)['B,1,VocabSize']:
        raise NotImplementedError

    def decoder_embedding(self,
                          src_ids: LongTensorT['B,SrcSeqLen'],
                          src_mask: IntTensorT['B,SrcSeqLen']):
        raise NotImplementedError


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
