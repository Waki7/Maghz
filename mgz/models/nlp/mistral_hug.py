""" PyTorch LED model."""
from __future__ import annotations

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

import mgz.settings as settings
from mgz.models.nlp.base_transformer import BaseTransformer, TransformerContext, \
    DecoderTransformer
from mgz.typing import *
from mgz.version_control.model_index import get_models_path


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
        hug.LlamaTokenizer]:
        try:
            return hug.LlamaTokenizer.from_pretrained(path)
        except (TypeError, FileNotFoundError, EnvironmentError) as e:
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


class PrototypeEmbedding(nn.Module):
    def __init__(self, n_tokens: int, hidden_size: int):
        super().__init__()
        # self.conv1 = nn.Conv1d(hidden_size, hidden_size, kernel_size=7,
        #                        padding=3, stride=5)
        self.embedder = nn.Linear(3 * hidden_size, hidden_size, bias=False)

    def forward(self, x: FloatTensorT['B,SrcSeqLen,EmbedLen']):
        # # treat embed len as the channels
        # x: FloatTensorT['B,EmbedLen,SrcSeqLen'] = x.permute(0, 2, 1)
        # # print('pre-conv', x.shape)
        # x = self.conv1(x)
        # x: FloatTensorT['B,SrcSeqLen/Stride,EmbedLen'] = x.permute(0, 2, 1)
        max_pool = torch.max(x, dim=1)
        avg_pool = torch.mean(x, dim=1)
        last = x[:, -1, :]
        combined_pool = torch.cat([max_pool.values, avg_pool, last], dim=-1)
        # print('post-conv', x.shape)
        # exit(3)
        x = self.embedder(combined_pool)
        return x


class MistralForCausalLMHug(MistralPreTrainedModel):
    FREEZE_CONFIGURATIONS = {
        "embed_only": ["model.encoder.embed_tokens",
                       "model.encoder.embed_positions", ],
        "embed_lm": ["model.encoder.embed_tokens",
                     "model.encoder.embed_positions", ],
        "embed_lm_last_decoder": ['embedding_head', 'lm_head',
                                  'model.layers.31'],
    }

    @classmethod
    def modules_to_not_convert(cls):
        return ['lm_head', 'embedding_head']

    def get_encoder(self):
        raise NotImplementedError

    def get_max_encoder_positions(self):
        raise NotImplementedError

    def get_decoder(self):
        raise NotImplementedError

    @overrides(BaseTransformer)
    def get_max_decoder_positions(self):
        return self.config.max_position_embeddings

    @overrides(BaseTransformer)
    def save(self, path: DirPath,
             quantization_config: Optional[BnbQuantizationConfig] = None,
             save_all_layers: bool = True):
        if not os.path.exists(path):
            os.makedirs(path)
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(self.config.to_dict(), f)
        weights_path: FilePath = os.path.normpath(
            os.path.join(path, 'embedding_head.bin'))
        torch.save(self.embedding_head.state_dict(), weights_path)

        if save_all_layers:
            weights_path: FilePath = os.path.normpath(
                os.path.join(path, 'weights.bin'))
            torch.save(self.hug.state_dict(), weights_path)

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
                # quantization_config.skip_modules = cls.modules_to_not_convert()
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
            model.hug.gradient_checkpointing = True
            return model
        except FileNotFoundError:
            return None

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        tokenizer = hug.LlamaTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.sep_token_id is None:
            tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.save_pretrained(path)
        if not os.path.exists(os.path.join(path, 'tokenizer.json')):
            with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
                json.dump(tokenizer.get_vocab(), f, indent=4)
        model_hug = hug.MistralForCausalLM.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
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
        torch.save(PrototypeEmbedding(
            model_hug.model.embed_tokens.num_embeddings,
            config.hidden_size).half().cpu().state_dict(),
                   os.path.normpath(os.path.join(path, 'embedding_head.bin')))

    def __init__(self, config: MistralConfig):
        super().__init__(config)
        torch.set_default_dtype(torch.float16)
        self.hug = hug.MistralForCausalLM(config)
        self.embedding_head = PrototypeEmbedding(
            self.hug.model.embed_tokens.num_embeddings,
            config.hidden_size)
        # self.post_embedding_layernorm = MistralRMSNorm(
        #     self.embedding_head.weight.shape[1], eps=config.rms_norm_eps)
        # Initialize weights and apply final processing
        self.apply(self._init_weights)
        # torch.set_default_dtype(torch.float32)

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
            lm_logits = self.hug.lm_head(output[:, -1, :])
        else:
            lm_logits = self.hug.lm_head(output)
        # embedding = self.embedding_head(output)

        max_pool = torch.max(output, dim=1)
        avg_pool = torch.mean(output, dim=1)
        last = output[:, -1, :]
        combined_pool = torch.cat([max_pool.values, avg_pool, last], dim=-1)

        return combined_pool, lm_logits

    @overrides(DecoderTransformer)
    def decode_relevance(self,
                         src_ids: LongTensorT['B,SrcSeqLen'],
                         src_mask: IntTensorT['B,SrcSeqLen'], ) -> \
            Tuple[FloatTensorT['B,EmbedLen'], Optional[
                FloatTensorT['B,NClasses']]]:
        full_output: BaseModelOutputWithPast = self.hug.model.forward(
            input_ids=src_ids, attention_mask=src_mask)
        output: FloatTensorT[
            'B,TgtSeqLen,EmbedLen'] = FloatTensorT(
            full_output.last_hidden_state)
        lm_logits = self.hug.lm_head(output[:, -2:, :])[:, -1, :]
        # unhinged wtf is this, these first two are not identical, the last two are
        # print(self.hug.lm_head(output[:, -1, :]))
        # print(self.hug.lm_head(output[:, -2:, :])[:,-1,:])
        # print(self.embedding_head(output[:, -1, :]))
        # print(self.embedding_head(output[:, -2:, :])[:,-1,:])
        embedding = self.embedding_head(output)
        # avg_embedding = output[:, -1, :]
        # print(avg_embedding.shape)
        # print(embedding.shape)
        # exit(3)
        return embedding, lm_logits

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
                                            max_new_tokens=max_new_tokens,
                                            do_sample=False)
        return LongTensorT(generate_output)


class MistralForCausalLMHugMock(MistralForCausalLMHug):

    @overrides(BaseTransformer)
    def save(self, path: DirPath,
             quantization_config: Optional[BnbQuantizationConfig] = None):
        pass

    @classmethod
    def load_model(cls, path: DirPath,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        MistralForCausalLMHug]:
        from accelerate import init_empty_weights
        try:
            with open(os.path.normpath(os.path.join(path, 'config.json')),
                      'r') as f:
                config = json.load(f)
            config = MistralConfig.from_dict(config)
            config._attn_implementation = "flash_attention_2"
            if torch.cuda.is_available():
                with init_empty_weights():
                    model = cls(config)
            else:
                model = cls(config)
            return model
        except FileNotFoundError:
            return None

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        tokenizer = hug.LlamaTokenizer.from_pretrained(model_id)
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token_id = tokenizer.eos_token_id
        if tokenizer.sep_token_id is None:
            tokenizer.sep_token_id = tokenizer.eos_token_id
        tokenizer.save_pretrained(path)
        if not os.path.exists(os.path.join(path, 'tokenizer.json')):
            with open(os.path.join(path, 'tokenizer.json'), 'w') as f:
                json.dump(tokenizer.get_vocab(), f, indent=4)
        config = hug.MistralConfig.from_pretrained(
            model_id,
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16, device_map={"": torch.device('cpu')}, )
        if config.pad_token_id is None:
            config.pad_token_id = tokenizer.pad_token_id
        if config.sep_token_id is None:
            config.sep_token_id = tokenizer.sep_token_id
        config._flash_attn_2_enabled = True
        config._attn_implementation = "flash_attention_2"

        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(config.to_dict(), f)

    @overrides(DecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,1|SrcSeqLen,SrcSeqLen'],
    ) -> LogitsTensorT['B,TgtSeqLen,OutNClasses']:
        b, src_len = src_ids.shape
        return LogitsTensorT(torch.rand(b, self.embed_dim))

    @overrides(DecoderTransformer)
    def decode_logits(self,
                      src_ids: LongTensorT['B,SrcSeqLen'],
                      src_mask: IntTensorT['B,SrcSeqLen'],
                      ret_last: bool = True) -> (
            LogitsTensorT)['B,Opt[SrcSeqLen],VocabSize']:
        b, src_len = src_ids.shape
        return LogitsTensorT(torch.rand(b, self.embed_dim))

    @overrides(DecoderTransformer)
    def decoder_embedding(self,
                          src_ids: LongTensorT['B,SrcSeqLen'],
                          src_mask: IntTensorT['B,SrcSeqLen'],
                          ret_last: bool = True) -> \
            FloatTensorT['B,Opt[SrcSeqLen],EmbedLen']:
        b, src_len = src_ids.shape
        return FloatTensorT(torch.rand(b, self.embed_dim))

    @overrides(DecoderTransformer)
    def decoder_embedding_w_logits(self,
                                   src_ids: LongTensorT['B,SrcSeqLen'],
                                   src_mask: IntTensorT['B,SrcSeqLen'],
                                   ret_last: bool = True) -> \
            Tuple[FloatTensorT['B,SrcSeqLen,EmbedLen'], FloatTensorT[
                'B,Opt[SrcSeqLen],VocabSize']]:
        b, src_len = src_ids.shape
        return FloatTensorT(torch.rand(b, self.embed_dim)), FloatTensorT(
            torch.rand(b, 2))

    @overrides(DecoderTransformer)
    def decode_relevance(self,
                         src_ids: LongTensorT['B,SrcSeqLen'],
                         src_mask: IntTensorT['B,SrcSeqLen'], ) -> \
            Tuple[FloatTensorT['B,EmbedLen'], Optional[
                FloatTensorT['B,NClasses']]]:
        b, src_len = src_ids.shape
        return FloatTensorT(torch.rand(b, self.embed_dim)), FloatTensorT(
            torch.rand(b, 2))

    @overrides(BaseTransformer)
    def generate(self,
                 src_ids: LongTensorT['B,SrcSeqLen'],
                 src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
                 tgt_ids: LongTensorT['B,TgtSeqLen'] = None,
                 max_new_tokens: int = None,
                 ) -> LongTensorT['TgtSeqLen']:
        b, src_len = src_ids.shape
        return LongTensorT(torch.randint(b, src_ids + 1))

    def __init__(self, config: MistralConfig):
        self.config = config
        self.embed_dim = config.hidden_size
        torch.nn.Module.__init__(self)


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
