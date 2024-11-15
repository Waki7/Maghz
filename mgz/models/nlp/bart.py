""" PyTorch BART model."""
from __future__ import annotations

import json
import math
import os
import random
from abc import ABC

import torch.utils.checkpoint
import transformers as hug
from accelerate import init_empty_weights
from accelerate.utils import load_and_quantize_model
from torch import nn
from transformers import BitsAndBytesConfig
from transformers.activations import ACT2FN
from transformers.models.bart.configuration_bart import BartConfig

import mgz.settings as settings
from mgz.models.nlp.base_transformer import BaseTransformer, TransformerContext, \
    EncoderDecoderTransformer
from mgz.models.nlp.utils_attention import _attention
from mgz.typing import *


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int,
                       decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    if pad_token_id is None:
        raise ValueError("self.model.config.pad_token_id has to be defined.")
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids


def _make_causal_mask(input_ids_shape: torch.Size, dtype: torch.dtype,
                      past_key_values_length: int = 0):
    """
    Make causal mask used for bi-directional self-attention.
    """
    bsz, tgt_len = input_ids_shape
    mask = torch.full((tgt_len, tgt_len), torch.tensor(torch.finfo(dtype).min))
    mask_cond = torch.arange(mask.size(-1))
    mask.masked_fill_(mask_cond < (mask_cond + 1).view(mask.size(-1), 1), 0)
    mask = mask.to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat(
            [torch.zeros(tgt_len, past_key_values_length, dtype=dtype), mask],
            dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


class BartLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        # Bart is set up so that if padding_idx is specified then offset the embedding ids by 2
        # and adjust num_embeddings appropriately. Other models don't have this hack
        self.offset = 2
        super().__init__(num_embeddings + self.offset, embedding_dim)

    def forward(self, input_ids: torch.Tensor,
                transformer_ctx: TransformerContext = None):
        """`input_ids' shape is expected to be [bsz x seqlen]."""
        past_key_values_length = transformer_ctx.get_seq_len_so_far() \
            if transformer_ctx is not None else 0
        bsz, seq_len = input_ids.shape[:2]
        positions = torch.arange(
            past_key_values_length, past_key_values_length + seq_len,
            dtype=torch.long, device=self.weight.device
        ).expand(bsz, -1)

        return super().forward(positions + self.offset)


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            decoder_attention: bool = False,
            dropout: ProbT = 0.0,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = num_heads
        self.decoder_attention = decoder_attention
        self.dropout = dropout
        self.head_dim = embed_dim // num_heads

        if (self.head_dim * num_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {num_heads})."
            )
        self.scaling = self.head_dim ** -0.5
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> \
            FloatTensorT['B,SrcSeqLen,NHeads']:
        return tensor.view(bsz, seq_len, self.n_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            # Embed Length must be divisible by num_heads
            query: FloatTensorT['B,TgtSeqLen,EmbedLen'],
            key: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            value: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            mask: IntTensorT['B,Opt[TgtSeqLen],TgtSeqLen'] = None,
            transformer_ctx: TransformerContext = None
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        """Input shape: Batch x Time x Channel"""
        "Implements Figure 2"
        if mask is not None and mask.dim() == 2:
            # Same mask applied to all tgt sequence if not specified
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        def n_head_reshape(tensor):
            return tensor.view(nbatches, -1, self.n_heads,
                               self.head_dim).transpose(
                1, 2).contiguous()

        # here we split the embedding into n_heads
        # 1) Do all the linear projections in batch from d_model => h x d_k
        query_st = n_head_reshape(self.q_proj(query)) * self.scaling
        if self.decoder_attention and transformer_ctx is not None and transformer_ctx.in_generation:
            key_st = transformer_ctx.add_key(
                n_head_reshape(self.k_proj(key)))
            value_st = transformer_ctx.add_value(n_head_reshape(
                self.v_proj(value)))
        else:
            key_st = n_head_reshape(self.k_proj(key))
            value_st = n_head_reshape(self.v_proj(value))

        proj_shape = (nbatches, self.n_heads, -1, self.head_dim)
        query_st = query_st.view(*proj_shape)
        key_st = key_st.view(*proj_shape)
        value_st = value_st.view(*proj_shape)

        # 2) Apply attention on all the projected vectors in batch.
        x = _attention(
            query_st, key_st, value_st, mask=mask,
            dropout_p=self.dropout if self.training else None,
        )

        # 3) "Concat" using a view and apply a final linear.
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatches, -1, self.n_heads * self.head_dim)
        )
        del query
        del key
        del value
        settings.empty_cache()
        return self.out_proj(x)


class BartEncoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = MultiHeadedAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout if self.training else None,
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            src_mask: IntTensorT['B,SrcSeqLen'],
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        residual = hidden_states

        hidden_states = self.self_attn.forward(
            query=hidden_states, key=hidden_states, value=hidden_states,
            mask=src_mask
        )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.activation_dropout,
                                              training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)

        if hidden_states.dtype == torch.float16 and (
                torch.isinf(hidden_states).any() or torch.isnan(
            hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value,
                                        max=clamp_value)
        return hidden_states


class BartDecoderLayer(nn.Module):
    def __init__(self, config: BartConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.d_model

        self.self_attn = MultiHeadedAttention(
            embed_dim=self.embed_dim,
            decoder_attention=True,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout if self.training else None,
        )
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout

        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.encoder_attn = MultiHeadedAttention(
            embed_dim=self.embed_dim,
            num_heads=config.decoder_attention_heads,
            dropout=config.attention_dropout if self.training else None,
        )
        self.encoder_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.fc1 = nn.Linear(self.embed_dim, config.decoder_ffn_dim)
        self.fc2 = nn.Linear(config.decoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    def forward(
            self,
            hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            encoder_hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            src_mask: IntTensorT['B,SrcSeqLen,TgtSeqLen'],
            tgt_mask: IntTensorT['B,SrcSeqLen,SrcSeqLen'], transformer_ctx=None
    ) -> FloatTensorT['B,TgtSeqLen|1,EmbedLen']:
        residual = hidden_states
        hidden_states = \
            self.self_attn.forward(
                query=hidden_states, key=hidden_states,
                value=hidden_states,
                mask=tgt_mask, transformer_ctx=transformer_ctx
            )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.self_attn_layer_norm(hidden_states)

        # Cross-Attention Block
        residual = hidden_states
        hidden_states = \
            self.encoder_attn.forward(
                query=hidden_states, key=encoder_hidden_states,
                value=encoder_hidden_states,
                mask=src_mask, transformer_ctx=transformer_ctx
            )
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.encoder_attn_layer_norm(hidden_states)

        # Fully Connected
        residual = hidden_states
        hidden_states = self.activation_fn(self.fc1(hidden_states))
        hidden_states = nn.functional.dropout(hidden_states,
                                              p=self.activation_dropout,
                                              training=self.training)
        hidden_states = self.fc2(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        hidden_states = residual + hidden_states
        hidden_states = self.final_layer_norm(hidden_states)
        return hidden_states


class BartPretrainedModel(EncoderDecoderTransformer, ABC):

    @classmethod
    def load_tokenizer(cls, tokenizer_id: str) -> Optional[
        hug.BartTokenizerFast]:
        try:
            return hug.BartTokenizerFast.from_pretrained(tokenizer_id)
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
        std = self.config.init_std
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    @property
    def dummy_inputs(self):
        pad_token = self.config.pad_token_id
        input_ids = torch.tensor([[0, 6, 10, 4, 2], [0, 8, 12, 2, pad_token]])
        dummy_inputs = {
            "attention_mask": input_ids.ne(pad_token),
            "input_ids": input_ids,
        }
        return dummy_inputs


class BartEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`BartEncoderLayer`].

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig,
                 embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(
            embed_dim) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim,
                                         self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [BartEncoderLayer(config) for _ in range(config.encoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(embed_dim)

        self.gradient_checkpointing = False

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,SrcSeqLen']
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        r"""
        Args:
            src_ids (`torch.LongTensor` of shape `(batch_size, sequence_length)`):
                Indices of input sequence tokens in the vocabulary. Padding will be ignored by default should you
                provide it.

                Indices can be obtained using [`AutoTokenizer`]. See [`PreTrainedTokenizer.encode`] and
                [`PreTrainedTokenizer.__call__`] for details.

                [What are input IDs?](../glossary#input-ids)
            src_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
                Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

                - 1 for tokens that are **not masked**,
                - 0 for tokens that are **masked**.

                [What are attention masks?](../glossary#attention-mask)
        """
        inputs_embeds = self.embed_tokens(src_ids) * self.embed_scale
        embed_pos = self.embed_positions(src_ids)
        embed_pos = embed_pos.to(inputs_embeds.device)
        hidden_states = inputs_embeds + embed_pos
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        encoder_layer: BartEncoderLayer
        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                    dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = None
            else:
                layer_outputs = encoder_layer.forward(
                    hidden_states=hidden_states,
                    src_mask=src_mask
                )

            hidden_states = layer_outputs
        return hidden_states


class BartDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`BartDecoderLayer`]

    Args:
        config: BartConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: BartConfig,
                 embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_position_embeddings
        self.embed_scale = math.sqrt(
            config.d_model) if config.scale_embedding else 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = BartLearnedPositionalEmbedding(
            config.max_position_embeddings,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [BartDecoderLayer(config) for _ in range(config.decoder_layers)])
        self.layernorm_embedding = nn.LayerNorm(config.d_model)

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def forward(
            self,
            encoder_hidden_states: FloatTensorT['B,SeqLen,EmbedLen'],
            tgt_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,1,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqStep'] = None,
            transformer_ctx: TransformerContext = None
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        inputs_embeds: FloatTensorT['B,1,EmbedLen'] = self.embed_tokens(
            tgt_ids) * self.embed_scale
        positions = self.embed_positions(tgt_ids, transformer_ctx)
        positions = positions.to(inputs_embeds.device)
        hidden_states = inputs_embeds + positions
        hidden_states = self.layernorm_embedding(hidden_states)
        hidden_states = nn.functional.dropout(hidden_states, p=self.dropout,
                                              training=self.training)
        # decoder layers
        decoder_layer: BartDecoderLayer
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            layer_outputs = decoder_layer.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                transformer_ctx=transformer_ctx.for_layer(
                    idx) if transformer_ctx else None
            )
            hidden_states = layer_outputs
        return hidden_states


class BartModel(BartPretrainedModel):

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_hug = hug.BartModel.from_pretrained(
            model_id).to(settings.DEVICE)
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(model_hug.config.to_dict(), f)
        torch.save(model_hug.state_dict(),
                   os.path.normpath(os.path.join(path, 'weights.bin')))
        tokenizer = hug.BartTokenizerFast.from_pretrained(model_id)
        tokenizer.save_pretrained(path)

    def __init__(self, config: BartConfig):
        super().__init__(config)

        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = BartEncoder(config, self.shared)
        self.decoder = BartDecoder(config, self.shared)
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'] = None
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        encoder_hidden_states = self.encode(src_ids=src_ids, src_mask=src_mask)
        decoder_hidden_state = self.decoder.forward(
            encoder_hidden_states=encoder_hidden_states,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return decoder_hidden_state


class BartForConditionalGeneration(BartPretrainedModel):
    @classmethod
    def load_model(cls, path: str,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        BartModel]:
        try:
            with open(os.path.normpath(os.path.join(path, 'config.json')),
                      'r') as f:
                config = json.load(f)
            with init_empty_weights():
                model = BartForConditionalGeneration(
                    hug.BartConfig.from_dict(config))
            weight_path = os.path.join(path, 'weights.bin')
            if not os.path.exists(weight_path):
                return None
            model = load_and_quantize_model(model,
                                            weights_location=weight_path,
                                            bnb_quantization_config=quantization_config,
                                            device_map="auto")
            assert isinstance(model, BartModel)
            return model
        except FileNotFoundError:
            return None

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_hug = hug.BartForConditionalGeneration.from_pretrained(
            model_id).to(settings.DEVICE)
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(model_hug.config.to_dict(), f)
        weight_path = os.path.normpath(os.path.join(path, 'weights.bin'))
        torch.save(model_hug.state_dict(), weight_path)
        logging.info('Saved model to %s', weight_path)
        tokenizer = hug.BartTokenizerFast.from_pretrained(model_id)
        tokenizer.save_pretrained(path)

    def __init__(self, config: BartConfig):
        super().__init__(config)
        self.model = BartModel(config)
        self.register_buffer("final_logits_bias",
                             torch.zeros((1, self.model.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model,
                                 self.model.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    def get_encoder(self):
        return self.model.get_encoder()

    def get_decoder(self):
        return self.model.get_decoder()

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        new_embeddings = super().resize_token_embeddings(new_num_tokens)
        self._resize_final_logits_bias(new_num_tokens)
        return new_embeddings

    def _resize_final_logits_bias(self, new_num_tokens: int) -> None:
        old_num_tokens = self.final_logits_bias.shape[-1]
        if new_num_tokens <= old_num_tokens:
            new_bias = self.final_logits_bias[:, :new_num_tokens]
        else:
            extra_bias = torch.zeros((1, new_num_tokens - old_num_tokens),
                                     device=self.final_logits_bias.device)
            new_bias = torch.cat([self.final_logits_bias, extra_bias], dim=1)
        self.register_buffer("final_logits_bias", new_bias)

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings

    @overrides(EncoderDecoderTransformer)
    def encoder_decoder_embedding(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'],
            pred_eos: bool = True,
    ) -> FloatTensorT['B,EmbedLen']:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.model.forward(
            src_ids=src_ids, src_mask=src_mask,
            tgt_ids=tgt_ids, tgt_mask=tgt_mask)

        if pred_eos:
            # get the last eos embedding
            sequence_lengths = \
                (torch.eq(tgt_ids, self.config.eos_token_id).long().argmax(
                    -1) - 1)
            bsz = output.shape[0]
            output_eos = output[torch.arange(bsz), sequence_lengths, :]
            return output_eos

        return output[:, -1, :]

    @overrides(EncoderDecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen']
    ) -> FloatTensorT['B,TgtSeqLen,OutNClasses']:
        r"""
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        Returns:
        """
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.model.forward(
            src_ids=src_ids, src_mask=src_mask,
            tgt_ids=tgt_ids, tgt_mask=tgt_mask)
        lm_logits = self.lm_head(output)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits

    @overrides(BaseTransformer)
    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        return self.model.encoder.forward(
            src_ids=src_ids,
            src_mask=src_mask,
        )

    @overrides(BaseTransformer)
    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1'],
               src_mask: IntTensorT['B,SrcSeqLen'] = None) -> \
            LogitsTensorT['B,1,VocabSize']:
        output = self.model.decoder.forward(
            encoder_hidden_states=transformer_ctx.encoder_memory,
            tgt_ids=generation_ids,
            src_mask=src_mask,
            transformer_ctx=transformer_ctx
        )
        lm_logits = self.lm_head(output)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits

    def prepare_decoder_input_ids_from_labels(self, labels: torch.Tensor):
        return shift_tokens_right(labels, self.config.pad_token_id,
                                  self.config.decoder_start_token_id)

    @staticmethod
    def _reorder_cache(past, beam_idx):
        reordered_past = ()
        for layer_past in past:
            # cached cross_attention states don't have to be reordered -> they are always the same
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in
                      layer_past[:2]) + layer_past[2:],
            )
        return reordered_past

