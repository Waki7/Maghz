""" PyTorch LED model."""
from __future__ import annotations

import inspect
import json
import os
from abc import ABC
from pathlib import Path

import torch.utils.checkpoint
import transformers as hug
from accelerate.utils import BnbQuantizationConfig
from torch import nn
from transformers import BitsAndBytesConfig, MistralConfig
from transformers.activations import ACT2FN
from transformers.modeling_outputs import BaseModelOutputWithPast
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

    @staticmethod
    def get_embedding_head(n_tokens: int, hidden_size: int):
        return nn.Linear(hidden_size, hidden_size, bias=False)
        # return nn.Linear(hidden_size, hidden_size, bias=False)

    @classmethod
    def modules_to_not_convert(cls):
        return ['lm_head', 'embedding_head']

    def get_encoder(self):
        raise NotImplementedError

    def get_max_encoder_positions(self):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    def get_max_decoder_positions(self):
        return self.hug.model.embed_tokens.weight.shape[1]

    @overrides(BaseTransformer)
    def save(self, path: DirPath,
             quantization_config: Optional[BnbQuantizationConfig] = None):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(self.config.to_dict(), f)
        weights_path: FilePath = os.path.normpath(
            os.path.join(path, 'embedding_head.bin'))
        torch.save(self.embedding_head.state_dict(), weights_path)

    @classmethod
    def load_model(cls, path: DirPath,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        MistralForCausalLMHug]:
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
            config = MistralConfig.from_dict(config)
            config._attn_implementation = "flash_attention_2"
            if torch.cuda.is_available():
                with init_empty_weights():
                    model = cls(config).half()
            else:
                model = cls(config).half()

            weight_path = os.path.join(path, 'weights.bin')
            embedding_weight_path = os.path.join(path, 'embedding_head.bin')
            if not os.path.exists(weight_path):
                weight_path = os.path.join(Path(path).parent.absolute(),
                                           'weights.bin')
                if not os.path.exists(weight_path):
                    weight_path = os.path.join(
                        Path(path).parent.parent.absolute(),
                        'weights.bin')

            if not os.path.exists(weight_path) or not os.path.exists(
                    embedding_weight_path):
                return None
            if quantization_config is not None:
                quantization_config.skip_modules = cls.modules_to_not_convert()
                model.hug = load_and_quantize_model(model.hug,
                                                    weights_location=weight_path,
                                                    bnb_quantization_config=quantization_config,
                                                    device_map="auto")
                model.embedding_head = load_and_quantize_model(
                    model.embedding_head,
                    weights_location=embedding_weight_path,
                    bnb_quantization_config=quantization_config,
                    device_map="auto")

            else:
                model.hug = load_checkpoint_and_dispatch(model.hug,
                                                         checkpoint=weight_path,
                                                         device_map={
                                                             "": settings.DEVICE})

                model.embedding_head = load_checkpoint_and_dispatch(
                    model.embedding_head,
                    checkpoint=embedding_weight_path,
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
        config = model_hug.config
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        if config.sep_token_id is None:
            config.sep_token_id = tokenizer.sep_token_id
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"

        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(config.to_dict(), f)
        torch.save(model_hug.state_dict(),
                   os.path.normpath(os.path.join(path, 'weights.bin')))
        torch.save(cls.get_embedding_head(
            model_hug.model.embed_tokens.num_embeddings,
            config.hidden_size).half().cpu().state_dict(),
                   os.path.normpath(os.path.join(path, 'embedding_head.bin')))

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        torch.set_default_dtype(torch.float16)
        self.hug = hug.MistralForCausalLM(config)
        self.embedding_head = self.get_embedding_head(
            self.hug.model.embed_tokens.num_embeddings,
            config.hidden_size)
        # self.post_embedding_layernorm = MistralRMSNorm(
        #     self.embedding_head.weight.shape[1], eps=config.rms_norm_eps)
        # Initialize weights and apply final processing
        self.apply(self._init_weights)
        torch.set_default_dtype(torch.float32)

    @overrides(DecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,1|SrcSeqLen,SrcSeqLen'],
    ) -> FloatTensorT['B,TgtSeqLen,OutNClasses']:
        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)
        lm_logits = self.hug.lm_head(output)
        return lm_logits

    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1'],
               src_mask: IntTensorT['B,SrcSeqLen'] = None) -> \
            LogitsTensorT['B,1,VocabSize']:
        r"""
        Should just be used for generation with torch.no_grad()
        """
        raise NotImplementedError

    def _change_output_if_configured(self, output: FloatTensorT[
        'B,Opt[SrcSeqLen],EmbedLen']) -> FloatTensorT[
        'B,Opt[SrcSeqLen],EmbedLen+2']:
        if self.embedding_head.weight.shape[1] == self.embed_dim + 2:
            if len(output.shape) == 2:
                output = FloatTensorT(
                    torch.cat([output, torch.zeros_like(output[:, :2])],
                              dim=-1))
            else:
                output = FloatTensorT(
                    torch.cat([output, torch.zeros_like(output[:, :, :2])],
                              dim=-1))
        return output

    @overrides(DecoderTransformer)
    def decode_logits(self,
                      src_ids: LongTensorT['B,SrcSeqLen'],
                      src_mask: IntTensorT['B,SrcSeqLen'],
                      ret_last: bool = True) -> (
            LogitsTensorT)['B,Opt[SrcSeqLen],VocabSize']:
        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)
        if ret_last:
            output = output[:, -1, :]
        output = self._change_output_if_configured(output)
        lm_logits = self.hug.lm_head(output)
        return lm_logits

    @overrides(DecoderTransformer)
    def decoder_embedding(self,
                          src_ids: LongTensorT['B,SrcSeqLen'],
                          src_mask: IntTensorT['B,SrcSeqLen'],
                          ret_last: bool = True) -> \
            FloatTensorT['B,Opt[SrcSeqLen],EmbedLen']:
        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)
        if ret_last:
            output = output[:, -1, :]
        output = self._change_output_if_configured(output)
        embedding = self.embedding_head(output)
        return embedding

    @overrides(DecoderTransformer)
    def decoder_embedding_w_logits(self,
                                   src_ids: LongTensorT['B,SrcSeqLen'],
                                   src_mask: IntTensorT['B,SrcSeqLen'],
                                   ret_last: bool = True) -> \
            Tuple[FloatTensorT['B,SrcSeqLen,EmbedLen'], FloatTensorT[
                'B,Opt[SrcSeqLen],VocabSize']]:
        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)
        if ret_last:
            output = output[:, -1, :]
        output = self._change_output_if_configured(output)
        lm_logits = self.hug.lm_head(output)
        embedding = self.embedding_head(output)
        return embedding, lm_logits

    @overrides(DecoderTransformer)
    def decode_relevance(self,
                         src_ids: LongTensorT['B,SrcSeqLen'],
                         src_mask: IntTensorT['B,SrcSeqLen']) -> \
            Tuple[FloatTensorT['B,EmbedLen'], Optional[FloatTensorT['B,2']]]:
        def get_llama_no_yes_scores(logits: FloatTensorT['NQuery,NClasses']):
            assert logits.shape[-1] == 32002
            Yes_id = 5613
            block_Yes_id = 5592
            yes_id = 9780
            block_yes_id = 5081
            yes_ids = [Yes_id, block_Yes_id, yes_id, block_yes_id]

            NO_id = 4032
            block_NO_id = 7929
            no_id = 1510
            block_no_id = 708
            No_id = 2501
            block_No_id = 1770
            no_ids = [NO_id, block_NO_id, no_id, block_no_id, No_id,
                      block_No_id]

            no_score = torch.max(logits[:, no_ids], dim=-1)[0]
            yes_score = torch.max(logits[:, yes_ids], dim=-1)[0]
            similarity_to_classes = FloatTensorT(
                torch.stack([no_score, yes_score], dim=-1))
            return similarity_to_classes

        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)[:, -1, :]
        lm_logits = self.hug.lm_head(output.detach())
        no_yes_logits = get_llama_no_yes_scores(lm_logits)
        embedding = self.embedding_head(output)
        return embedding, no_yes_logits

    @overrides(BaseTransformer)
    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,TgtSeqLen'] = None,
                 max_new_tokens: int = None,
                 ) -> LongTensorT['TgtSeqLen']:
        assert src_ids.shape[
                   -1] - max_new_tokens < self.get_max_decoder_positions(), \
            'TODO, find exact mechanism that triggers the stop'
        generate_output = self.hug.generate(src_ids,
                                            attention_mask=src_mask,
                                            max_new_tokens=max_new_tokens, )
        return LongTensorT(generate_output)


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
