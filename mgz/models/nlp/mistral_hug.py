""" PyTorch LED model."""
from __future__ import annotations

import inspect
import json
import os
from abc import ABC

import torch.utils.checkpoint
import transformers as hug
from torch import nn
from transformers import BitsAndBytesConfig, MistralConfig
from transformers.activations import ACT2FN
from transformers.utils.import_utils import is_flash_attn_2_available

import mgz.settings as settings
from mgz.models.nlp.base_transformer import BaseTransformer, TransformerContext, \
    DecoderTransformer
from mgz.typing import *
from mgz.version_control.model_index import get_models_path

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters)


class MistralMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.embed_dim, self.intermediate_size,
                                   bias=False)
        self.up_proj = nn.Linear(self.embed_dim, self.intermediate_size,
                                 bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.embed_dim,
                                   bias=False)
        self.act_fn = ACT2FN[config.hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(batch,
                                                           num_key_value_heads,
                                                           n_rep, slen,
                                                           head_dim)
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen,
                                 head_dim)


class MistralPreTrainedModel(DecoderTransformer, ABC):

    @classmethod
    def modules_to_apply_lora(cls) -> List[str]:
        return ['key', 'value', 'query']

    @classmethod
    def modules_to_not_convert(cls):
        return ['model.encoder.embed_tokens', 'model.decoder.embed_tokens',
                'lm_head']

    @classmethod
    def load_tokenizer(cls, path: DirPath) -> Optional[
        hug.LlamaTokenizerFast]:
        try:
            return hug.LlamaTokenizerFast.from_pretrained(path)
        except (FileNotFoundError, EnvironmentError) as e:
            return None

    @overrides(BaseTransformer)
    def save(self, path: DirPath):
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(self.config.to_dict(), f)
        torch.save(self.state_dict(),
                   os.path.normpath(os.path.join(path, 'weights.bin')))

    def _init_weights(self, module):
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


class MistralForCausalLMHug(MistralPreTrainedModel):

    @classmethod
    def modules_to_not_convert(cls):
        return ['model.encoder.embed_tokens', 'model.decoder.embed_tokens',
                'lm_head' + 'embedding_head']

    def get_encoder(self):
        raise NotImplementedError

    def get_max_encoder_positions(self):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    def get_max_decoder_positions(self):
        return self.model.embed_tokens.weight.shape[1]

    @classmethod
    def load_model(cls, path: str,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        MistralPreTrainedModel]:
        from accelerate import init_empty_weights, load_checkpoint_and_dispatch
        from accelerate.utils import load_and_quantize_model
        try:
            with open(os.path.normpath(os.path.join(path, 'config.json')),
                      'r') as f:
                config = json.load(f)
                if hasattr(config, 'quantization_config'):
                    if (quantization_config != config.quantization_config):
                        logging.warning(
                            'quantization configs do not match, {} vs {}'.format(
                                quantization_config,
                                config.quantization_config))
                    quantization_config = config.quantization_config
            with init_empty_weights():
                model = MistralForCausalLMHug(
                    MistralConfig.from_dict(config)).half()
            if not os.path.exists(os.path.join(path, 'weights.bin')):
                return None
            if quantization_config:
                quantization_config.llm_int8_skip_modules = cls.modules_to_not_convert()
                model = load_and_quantize_model(model,
                                                weights_location=os.path.join(
                                                    path,
                                                    'weights.bin'),
                                                bnb_quantization_config=quantization_config,
                                                device_map={
                                                    "": settings.DEVICE})
            else:
                model = load_checkpoint_and_dispatch(model,
                                                     checkpoint=os.path.join(
                                                         path,
                                                         'weights.bin'),
                                                     device_map={
                                                         "": settings.DEVICE})
            assert isinstance(model, MistralForCausalLMHug)
            return model
        except FileNotFoundError:
            return None

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        tokenizer = hug.LlamaTokenizerFast.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.sep_token_id is None:
            tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.save_pretrained(path)
        model_hug = hug.MistralForCausalLM.from_pretrained(
            model_id,
            use_flash_attention_2=True,
            torch_dtype=torch.float16, device_map={"": torch.device('cpu')}, )
        model_hug.add_module('embedding_head', nn.Linear(
            model_hug.config.hidden_size,
            model_hug.config.hidden_size,
            bias=False))
        config = model_hug.config
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        if config.sep_token_id is None:
            config.sep_token_id = tokenizer.sep_token_id
        config._flash_attn_2_enabled = True
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(model_hug.config.to_dict(), f)
        torch.save(model_hug.state_dict(),
                   os.path.normpath(os.path.join(path, 'weights.bin')))

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        config._flash_attn_2_enabled = True
        self.model = hug.MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.model.embed_tokens.num_embeddings,
                                 # 4096,
                                 bias=False)
        self.embedding_head = nn.Linear(config.hidden_size,
                                        config.hidden_size,
                                        bias=False)
        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    @overrides(DecoderTransformer)
    def decoder_embedding(
            self,
            src_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,TgtSeqLen'],
            pred_eos: bool = False,
    ) -> FloatTensorT['B,EmbedLen']:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=torch.LongTensor(src_ids),
            attention_mask=src_ids,
            output_hidden_states=True,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.embedding_head(hidden_states)
        return FloatTensorT(logits)

    @overrides(DecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,1|SrcSeqLen,SrcSeqLen'],
    ) -> FloatTensorT['B,TgtSeqLen,OutNClasses']:
        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model.forward(
            input_ids=torch.LongTensor(src_ids),
            attention_mask=src_ids,
            output_hidden_states=True,
            return_dict=False,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        # return FloatTensorT(CausalLMOutputWithPast(
        #     logits=logits,
        #     past_key_values=outputs.past_key_values,
        #     hidden_states=outputs.hidden_states,
        #     attentions=outputs.attentions,
        # ).logits)
        return FloatTensorT(logits)

    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1'],
               src_mask: IntTensorT['B,SrcSeqLen'] = None) -> \
            LogitsTensorT['B,1,VocabSize']:
        raise NotImplementedError

    @overrides(BaseTransformer)
    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,TgtSeqLen'] = None,
                 max_new_tokens: int = None,
                 ) -> LongTensorT['TgtSeqLen']:
        return LongTensorT(
            self.model.generate(input_ids=src_ids, attention_mask=src_mask,
                                max_new_tokens=max_new_tokens))


def main():
    pth = os.path.join(get_models_path(), 'mistralai/Mistral-7B-v0.1')
    model = hug.MistralForSequenceClassification(
        hug.MistralConfig())
    model.load_state_dict(
        torch.load(pth + '/pytorch_model-00001-of-00002.bin', ), strict=False)
    model.load_state_dict(
        torch.load(pth + '/pytorch_model-00002-of-00002.bin', ), strict=False)
    torch.save(model.half().cuda().state_dict(), pth + '/weights.bin')


if __name__ == '__main__':
    main()
