""" PyTorch LED model."""
from __future__ import annotations

import inspect
import json
import os
from abc import ABC
from pathlib import Path

import torch.nn.functional as F
import torch.utils.checkpoint
import transformers as hug
from accelerate.utils import BnbQuantizationConfig
from torch import nn
from transformers import BitsAndBytesConfig, MistralConfig
from transformers.activations import ACT2FN
from transformers.utils.import_utils import is_flash_attn_2_available

import mgz.settings as settings
from mgz.models.nlp.base_transformer import BaseTransformer, TransformerContext, \
    DecoderTransformer, InferenceContext
from mgz.models.nlp.utils_attention import _attention
from mgz.typing import *
from mgz.version_control.model_index import get_models_path

if is_flash_attn_2_available():
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    from flash_attn.bert_padding import index_first_axis, pad_input, \
        unpad_input  # noqa

    _flash_supports_window_size = "window_size" in list(
        inspect.signature(flash_attn_func).parameters)


# Copied from transformers.models.llama.modeling_llama._get_unpad_data
def _get_unpad_data(padding_mask):
    seqlens_in_batch = padding_mask.sum(dim=-1, dtype=torch.int32)
    indices = torch.nonzero(padding_mask.flatten(), as_tuple=False).flatten()
    max_seqlen_in_batch = seqlens_in_batch.max().item()
    cu_seqlens = F.pad(
        torch.cumsum(seqlens_in_batch, dim=0, dtype=torch.torch.int32), (1, 0))
    return (
        indices,
        cu_seqlens,
        max_seqlen_in_batch,
    )


def _make_sliding_window_causal_mask(
        input_ids_shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
        past_key_values_length: int = 0,
        sliding_window: int = 4096,
):
    """
    Make causal mask used for sliding window attention
    """
    bsz, tgt_len = input_ids_shape

    tensor = torch.full(
        (tgt_len, tgt_len),
        fill_value=1,
        device=device,
    )
    mask = torch.tril(tensor, diagonal=0)
    # make the mask banded to account for sliding window
    mask = torch.triu(mask, diagonal=-sliding_window)
    mask = torch.log(mask).to(dtype)

    if past_key_values_length > 0:
        mask = torch.cat([torch.zeros(tgt_len, past_key_values_length,
                                      dtype=dtype, device=device), mask],
                         dim=-1)
    return mask[None, None, :, :].expand(bsz, 1, tgt_len,
                                         tgt_len + past_key_values_length)


# Copied from transformers.models.bart.modeling_bart._expand_mask
def _expand_mask(mask: torch.Tensor, dtype: torch.dtype,
                 tgt_len: Optional[int] = None):
    """
    Expands attention_mask from `[bsz, seq_len]` to `[bsz, 1, tgt_seq_len, src_seq_len]`.
    """
    bsz, src_len = mask.size()
    tgt_len = tgt_len if tgt_len is not None else src_len

    expanded_mask = mask[:, None, None, :].expand(bsz, 1, tgt_len, src_len).to(
        dtype)

    inverted_mask = 1.0 - expanded_mask

    return inverted_mask.masked_fill(inverted_mask.to(torch.bool),
                                     torch.finfo(dtype).min)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Mistral
class MistralRMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps=1e-6):
        """
        MistralRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(
            variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


# Copied from transformers.models.llama.modeling_llama.LlamaRotaryEmbedding with Llama->Mistral
class MistralRotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000,
                 device=None):
        super().__init__()

        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        inv_freq = 1.0 / (self.base ** (
                torch.arange(0, self.dim, 2).float().to(device) / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

        # Build here to make `torch.jit.trace` work.
        self._set_cos_sin_cache(
            seq_len=max_position_embeddings, device=self.inv_freq.device,
            dtype=torch.get_default_dtype()
        )

    def _set_cos_sin_cache(self, seq_len, device, dtype):
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device,
                         dtype=self.inv_freq.dtype)

        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached",
                             emb.cos()[None, None, :, :].to(dtype),
                             persistent=False)
        self.register_buffer("sin_cached",
                             emb.sin()[None, None, :, :].to(dtype),
                             persistent=False)

    def forward(self, x, seq_len=None):
        # x: [bs, num_attention_heads, seq_len, head_size]
        if seq_len > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=seq_len, device=x.device,
                                    dtype=x.dtype)

        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )


# Copied from transformers.models.llama.modeling_llama.rotate_half
def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


# Copied from transformers.models.llama.modeling_llama.apply_rotary_pos_emb
def apply_rotary_pos_emb(q, k, cos, sin, position_ids):
    # The first two dimensions of cos and sin are always 1, so we can `squeeze` them.
    cos = cos.squeeze(1).squeeze(0)  # [seq_len, dim]
    sin = sin.squeeze(1).squeeze(0)  # [seq_len, dim]
    cos = cos[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    sin = sin[position_ids].unsqueeze(1)  # [bs, 1, seq_len, dim]
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


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


class MistralAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            config: MistralConfig,
            is_decoder_self_attention: bool = False,
    ):
        super().__init__()
        self.config = config
        self.decoder_attention = is_decoder_self_attention

        embed_dim = config.hidden_size
        self.embed_dim = embed_dim
        self.n_heads = config.num_attention_heads
        self.head_dim = embed_dim // self.n_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.n_heads // self.num_key_value_heads
        self.max_position_embeddings = config.max_position_embeddings
        self.rope_theta = config.rope_theta

        if (self.head_dim * self.n_heads) != self.embed_dim:
            raise ValueError(
                f"embed_dim must be divisible by num_heads (got `embed_dim`: {self.embed_dim}"
                f" and `num_heads`: {self.n_heads})."
            )
        self.q_proj = nn.Linear(self.embed_dim,
                                self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(self.embed_dim,
                                self.num_key_value_heads * self.head_dim,
                                bias=False)
        self.v_proj = nn.Linear(self.embed_dim,
                                self.num_key_value_heads * self.head_dim,
                                bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim,
                                self.embed_dim, bias=False)

        self.rotary_emb = MistralRotaryEmbedding(
            self.head_dim,
            max_position_embeddings=self.max_position_embeddings,
            base=self.rope_theta,
        )

    def _shape(self, tensor: torch.Tensor, seq_len: int, bsz: int) -> \
            FloatTensorT['B,SrcSeqLen,NHeads']:
        return tensor.view(bsz, seq_len, self.n_heads,
                           self.head_dim).transpose(1, 2).contiguous()

    def forward(
            self,
            # Embed Length must be divisible by n_heads
            query: FloatTensorT['B,TgtSeqLen,EmbedLen'],
            key: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            value: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            position_ids: LongTensorT['TgtSeqStep|SrcSeqLen'],
            src_mask: IntTensorT['B,Opt[TgtSeqLen],TgtSeqLen'],
            padding_mask: IntTensorT['B,TgtSeqLen'] = None,
            transformer_ctx: TransformerContext = None
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        if src_mask is not None and src_mask.dim() == 2:
            # Same mask applied to all tgt sequence if not specified
            mask = src_mask.unsqueeze(1)
        nbatches, q_len, _ = query.size()

        query_st = self.q_proj(query)
        key_st = self.k_proj(key)
        value_st = self.v_proj(value)

        query_st = query_st.view(nbatches, q_len, self.n_heads,
                                 self.head_dim).transpose(1, 2)
        key_st = key_st.view(nbatches, q_len, self.num_key_value_heads,
                             self.head_dim).transpose(1, 2)
        value_st = value_st.view(nbatches, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)

        kv_seq_len = key_st.shape[-2]
        if transformer_ctx is not None:
            kv_seq_len += transformer_ctx.get_seq_len_so_far()
        cos, sin = self.rotary_emb(value_st, seq_len=kv_seq_len)
        query_st, key_st = (
            apply_rotary_pos_emb(query_st, key_st, cos, sin, position_ids))

        if (self.decoder_attention and transformer_ctx is not None
                and transformer_ctx.in_generation):
            key_st = transformer_ctx.add_key(
                (key.view(nbatches, q_len, self.num_key_value_heads,
                          self.head_dim).transpose(1, 2)))
            value_st = transformer_ctx.add_value(
                (value.view(nbatches, q_len, self.num_key_value_heads,
                            self.head_dim).transpose(1, 2)))

        # repeat k/v heads if n_kv_heads < n_heads
        key_st = FloatTensorT(repeat_kv(key_st, self.num_key_value_groups))
        value_st = FloatTensorT(repeat_kv(value_st, self.num_key_value_groups))

        # 2) Apply attention on all the projected vectors in batch.
        x = _attention(
            query_st, key_st, value_st, mask=mask,
            dropout_p=None,
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
        return self.o_proj(x)


class MistralFlashAttention2(MistralAttention):
    """
    Mistral flash attention module. This module inherits from `MistralAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def forward(
            self,
            query: FloatTensorT['B,TgtSeqLen,EmbedLen'],
            key: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            value: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            position_ids: LongTensorT['TgtSeqStep|SrcSeqLen'],
            src_mask: IntTensorT['B,Opt[TgtSeqLen],TgtSeqLen'],
            padding_mask: IntTensorT['B,TgtSeqLen'] = None,
            transformer_ctx: TransformerContext = None
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        nbatches, q_len, _ = query.size()

        query_st = self.q_proj(query)
        key_st = self.k_proj(key)
        value_st = self.v_proj(value)

        query_st = query_st.view(nbatches, q_len, self.n_heads,
                                 self.head_dim).transpose(1, 2)
        key_st = key_st.view(nbatches, q_len, self.num_key_value_heads,
                             self.head_dim).transpose(1, 2)
        value_st = value_st.view(nbatches, q_len, self.num_key_value_heads,
                                 self.head_dim).transpose(1, 2)

        kv_seq_len = key_st.shape[-2]
        if transformer_ctx is not None:
            kv_seq_len += transformer_ctx.get_seq_len_so_far()

        # Because the input can be padded, the absolute sequence length depends on the max position id.
        rotary_seq_len = max(kv_seq_len, position_ids[:, -1].max().item()) + 1
        cos, sin = self.rotary_emb(value_st, seq_len=rotary_seq_len)

        query_st, key_st = apply_rotary_pos_emb(query_st,
                                                key_st, cos, sin,
                                                position_ids)

        use_sliding_windows = (
                _flash_supports_window_size
                and hasattr(self.config, "sliding_window") is not None
                and kv_seq_len > self.config.sliding_window
        )

        if not _flash_supports_window_size:
            logging.warning_once(
                "The current flash attention version does not support sliding window attention, for a more memory efficient implementation"
                " make sure to upgrade flash-attn library."
            )
        if (transformer_ctx is not None and transformer_ctx.in_generation and
                transformer_ctx.get_seq_len_so_far() > 0):
            past_key = transformer_ctx.get_key()
            past_value = transformer_ctx.get_value()
            # Activate slicing cache only if the config has a value `sliding_windows` attribute
            if hasattr(self.config,
                       "sliding_window") and kv_seq_len > self.config.sliding_window:
                slicing_tokens = kv_seq_len - self.config.sliding_window

                past_key = transformer_ctx.get_key()
                past_value = transformer_ctx.get_value()

                past_key = past_key[:, :, slicing_tokens:, :].contiguous()
                past_value = past_value[:, :, slicing_tokens:, :].contiguous()
                if past_key.shape[-2] != self.config.sliding_window - 1:
                    raise ValueError(
                        f"past key much have a shape of (`batch_size, num_heads, self.config.sliding_window-1, head_dim`), got"
                        f" {past_key.shape}"
                    )
            key_st = FloatTensorT(torch.cat([past_key, key_st], dim=2))
            value_st = FloatTensorT(
                torch.cat([past_value, value_st], dim=2))

        if transformer_ctx is not None:
            key_st = transformer_ctx.set_key(key_st)
            value_st = transformer_ctx.set_value(value_st)

        # repeat k/v heads if n_kv_heads < n_heads
        key_st = FloatTensorT(repeat_kv(key_st, self.num_key_value_groups))
        value_st = FloatTensorT(repeat_kv(value_st, self.num_key_value_groups))

        # TODO: Mistral does not have dropout in the config??
        # It is recommended to use dropout with FA according to the docs
        # when training.
        dropout_rate = 0.0  # if not self.training else self.attn_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        query_st = query_st.to(torch.float16)
        key_st = key_st.to(torch.float16)
        value_st = value_st.to(torch.float16)

        # Reashape to the expected shape for Flash Attention
        query_st: FloatTensorT['B,NHeads,TgtSeqLen,EmbedLen'] = (
            query_st.transpose(1, 2))
        key_st: FloatTensorT['B,NHeads,TgtSeqLen,EmbedLen'] = (
            key_st.transpose(1, 2))
        value_st: FloatTensorT['B,NHeads,TgtSeqLen,EmbedLen'] = (
            value_st.transpose(1, 2))
        attn_output = self._flash_attention_forward(
            query_st,
            key_st,
            value_st,
            padding_mask,
            q_len,
            dropout=dropout_rate,
            use_sliding_windows=use_sliding_windows,
        )

        attn_output = attn_output.reshape(nbatches, q_len,
                                          self.embed_dim).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output

    def _flash_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            padding_mask,
            query_length,
            dropout=0.0,
            softmax_scale=None,
            use_sliding_windows=False,
    ):
        """
        Calls the forward method of Flash Attention - if the input hidden states contain at least one padding token
        first unpad the input, then computes the attention scores and pad the final attention scores.

        Args:
            query_states (`torch.Tensor`):
                Input query states to be passed to Flash Attention API
            key_states (`torch.Tensor`):
                Input key states to be passed to Flash Attention API
            value_states (`torch.Tensor`):
                Input value states to be passed to Flash Attention API
            padding_mask (`torch.Tensor`):
                The padding mask - corresponds to a tensor of size `(batch_size, seq_len)` where 0 stands for the
                position of padding tokens and 1 for the position of non-padding tokens.
            dropout (`int`, *optional*):
                Attention dropout
            softmax_scale (`float`, *optional*):
                The scaling of QK^T before applying softmax. Default to 1 / sqrt(head_dim)
            use_sliding_windows (`bool`, *optional*):
                Whether to activate sliding window attention.
        """
        # Contains at least one padding token in the sequence
        if padding_mask is not None:
            batch_size = query_states.shape[0]
            query_states, key_states, value_states, indices_q, cu_seq_lens, max_seq_lens = self._upad_input(
                query_states, key_states, value_states, padding_mask,
                query_length
            )

            cu_seqlens_q, cu_seqlens_k = cu_seq_lens
            max_seqlen_in_batch_q, max_seqlen_in_batch_k = max_seq_lens

            if not use_sliding_windows:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                )
            else:
                attn_output_unpad = flash_attn_varlen_func(
                    query_states,
                    key_states,
                    value_states,
                    cu_seqlens_q=cu_seqlens_q,
                    cu_seqlens_k=cu_seqlens_k,
                    max_seqlen_q=max_seqlen_in_batch_q,
                    max_seqlen_k=max_seqlen_in_batch_k,
                    dropout_p=dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(
                        self.config.sliding_window, self.config.sliding_window),
                )

            attn_output = pad_input(attn_output_unpad, indices_q, batch_size,
                                    query_length)
        else:
            if not use_sliding_windows:
                attn_output = flash_attn_func(
                    query_states, key_states, value_states, dropout,
                    softmax_scale=softmax_scale, causal=True
                )
            else:
                attn_output = flash_attn_func(
                    query_states,
                    key_states,
                    value_states,
                    dropout,
                    softmax_scale=softmax_scale,
                    causal=True,
                    window_size=(
                        self.config.sliding_window, self.config.sliding_window),
                )

        return attn_output

    def _upad_input(self, query_layer, key_layer, value_layer, padding_mask,
                    query_length):
        batch_size, kv_seq_len, num_heads, head_dim = key_layer.shape

        # On the first iteration we need to properly re-create the padding mask
        # by slicing it on the proper place
        if kv_seq_len != padding_mask.shape[-1]:
            padding_mask_num_tokens = padding_mask.shape[-1]
            padding_mask = padding_mask[:,
                           padding_mask_num_tokens - kv_seq_len:]

        indices_k, cu_seqlens_k, max_seqlen_in_batch_k = _get_unpad_data(
            padding_mask)
        key_layer = index_first_axis(
            key_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
            indices_k)
        value_layer = index_first_axis(
            value_layer.reshape(batch_size * kv_seq_len, num_heads, head_dim),
            indices_k)

        if query_length == kv_seq_len:
            query_layer = index_first_axis(
                query_layer.reshape(batch_size * kv_seq_len, num_heads,
                                    head_dim), indices_k
            )
            cu_seqlens_q = cu_seqlens_k
            max_seqlen_in_batch_q = max_seqlen_in_batch_k
            indices_q = indices_k
        elif query_length == 1:
            max_seqlen_in_batch_q = 1
            cu_seqlens_q = torch.arange(
                batch_size + 1, dtype=torch.int32, device=query_layer.device
            )  # There is a memcpy here, that is very bad.
            indices_q = cu_seqlens_q[:-1]
            query_layer = query_layer.squeeze(1)
        else:
            # The -q_len: slice assumes left padding.
            padding_mask = padding_mask[:, -query_length:]
            query_layer, indices_q, cu_seqlens_q, max_seqlen_in_batch_q = unpad_input(
                query_layer, padding_mask)

        return (
            query_layer,
            key_layer,
            value_layer,
            indices_q,
            (cu_seqlens_q, cu_seqlens_k),
            (max_seqlen_in_batch_q, max_seqlen_in_batch_k),
        )


class MistralDecoderLayer(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        self.config = config
        self.embed_dim = config.hidden_size
        self.self_attn = (
            MistralAttention(config)
            if not getattr(config, "_flash_attn_2_enabled", False)
            else MistralFlashAttention2(config)
        )

        self.mlp = MistralMLP(config)
        self.input_layernorm = MistralRMSNorm(config.hidden_size,
                                              eps=config.rms_norm_eps)
        self.post_attention_layernorm = MistralRMSNorm(config.hidden_size,
                                                       eps=config.rms_norm_eps)

    def forward(
            self,
            hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            position_ids: LongTensorT['TgtSeqStep|SrcSeqLen'],
            src_mask: IntTensorT['B,SrcSeqLen,SrcSeqLen'],
            padding_mask: IntTensorT['B,TgtSeqLen'],
            transformer_ctx: TransformerContext = None
    ) -> FloatTensorT['B,TgtSeqLen|1,EmbedLen']:
        residual = hidden_states

        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = \
            self.self_attn.forward(
                query=hidden_states, key=hidden_states,
                value=hidden_states,
                src_mask=src_mask,
                padding_mask=padding_mask,
                transformer_ctx=transformer_ctx,
                position_ids=position_ids
            )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states

        return hidden_states


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
        except (FileNotFoundError, EnvironmentError) as e:
            return None

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
        weights_path: FilePath = os.path.normpath(
            os.path.join(path, 'weights.bin'))
        core_model = self.state_dict()
        core_model.pop('embedding_head.weight')
        torch.save(core_model, weights_path)

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


class MistralModel(MistralPreTrainedModel):
    def __init__(self, config: MistralConfig):
        super().__init__(config)
        self.gradient_checkpointing = True
        self.config: MistralConfig = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.embed_tokens = nn.Embedding(vocab_size, config.hidden_size,
                                         padding_idx)
        self.layers = nn.ModuleList([MistralDecoderLayer(config) for _ in
                                     range(config.num_hidden_layers)])
        self.norm = MistralRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.apply(self._init_weights)

    def _prepare_decoder_attention_mask(
            self, attention_mask, input_shape, inputs_embeds,
            past_key_values_length, sliding_window
    ):
        # create causal mask
        # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
        combined_attention_mask = None
        if input_shape[-1] > 1:
            combined_attention_mask = _make_sliding_window_causal_mask(
                input_shape,
                inputs_embeds.dtype,
                device=inputs_embeds.device,
                past_key_values_length=past_key_values_length,
                sliding_window=sliding_window,
            )

        if attention_mask is not None:
            # [bsz, seq_len] -> [bsz, 1, tgt_seq_len, src_seq_len]
            expanded_attn_mask = _expand_mask(attention_mask,
                                              inputs_embeds.dtype,
                                              tgt_len=input_shape[-1]).to(
                inputs_embeds.device
            )
            combined_attention_mask = (
                expanded_attn_mask if combined_attention_mask is None else expanded_attn_mask + combined_attention_mask
            )

        return combined_attention_mask

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    @overrides(DecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,TgtSeqLen'],
            transformer_ctx: TransformerContext = None,
            src_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'] = None
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        src_ids, src_mask = self._pre_encode_pad_if_needed(
            src_ids=src_ids,
            src_mask=src_mask,
            pad_token_id=self.config.pad_token_id,
        )
        input_embeds = self.embed_tokens(src_ids)
        hidden_state = input_embeds

        padding_mask = src_mask
        if padding_mask is not None and (hasattr(self.config,
                                                 "_flash_attn_2_enabled") and self.config._flash_attn_2_enabled):
            is_padding_right = (
                    padding_mask[:, -1].sum().item() != src_ids.shape[0])
            if is_padding_right:
                raise ValueError(
                    "You are attempting to perform batched generation with padding_side='right'"
                    " this may lead to unexpected behaviour for Flash Attention version of Mistral. Make sure to "
                    " call `tokenizer.padding_side  = 'left'` before tokenizing the input. "
                )
        src_mask = self._prepare_decoder_attention_mask(
            src_mask,
            src_ids.shape,
            input_embeds,
            transformer_ctx.get_seq_len_so_far() if transformer_ctx else 0,
            sliding_window=self.config.sliding_window,
        )

        seq_length = src_ids.shape[1]
        prev_length = transformer_ctx.get_seq_len_so_far() if transformer_ctx else 0
        position_ids = torch.arange(
            prev_length, seq_length + prev_length,
            dtype=torch.long, device=src_ids.device
        )
        position_ids = position_ids.unsqueeze(0).view(-1, seq_length)
        decoder_layer: MistralDecoderLayer
        for idx, decoder_layer in enumerate(self.layers):
            if self.gradient_checkpointing and self.training:
                hidden_state = torch.utils.checkpoint.checkpoint(
                    decoder_layer,
                    hidden_state,
                    position_ids,
                    src_mask,
                    padding_mask,
                    transformer_ctx.for_layer(idx) if transformer_ctx else None,
                    use_reentrant=True,
                )
            else:
                hidden_state = decoder_layer.forward(hidden_states=hidden_state,
                                                     position_ids=position_ids,
                                                     src_mask=src_mask,
                                                     padding_mask=padding_mask,
                                                     transformer_ctx=transformer_ctx.for_layer(
                                                         idx) if transformer_ctx else None)

        hidden_state = self.norm(hidden_state)
        return hidden_state


class _MistralForCausalLM(nn.Module):
    def __init__(self, config: MistralConfig):
        super().__init__()
        config._flash_attn_2_enabled = True
        self.model = MistralModel(config)
        self.lm_head = nn.Linear(config.hidden_size,
                                 self.model.embed_tokens.num_embeddings,
                                 # 4096,
                                 bias=False)


class MistralForCausalLM(MistralPreTrainedModel):
    @staticmethod
    def get_embedding_head(n_tokens: int, hidden_size: int):
        # return nn.Linear(hidden_size + 2, hidden_size, bias=False)
        return nn.Linear(hidden_size, hidden_size, bias=False)

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

    @classmethod
    def load_model(cls, path: DirPath,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        MistralForCausalLM]:
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
            assert isinstance(model, MistralForCausalLM)
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
        config._flash_attn_2_enabled = True
        self.hug = _MistralForCausalLM(config)
        self.embedding_head = self.get_embedding_head(
            self.hug.model.embed_tokens.num_embeddings,
            config.hidden_size)
        # Initialize weights and apply final processing
        self.apply(self._init_weights)

    @overrides(DecoderTransformer)
    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,1|SrcSeqLen,SrcSeqLen'],
    ) -> FloatTensorT['B,TgtSeqLen,OutNClasses']:
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.hug.model.forward(
            src_ids=src_ids, src_mask=src_mask)
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
        output: FloatTensorT['B,1,EmbedLen'] = self.hug.model.forward(
            src_ids=generation_ids, src_mask=src_mask,
            transformer_ctx=transformer_ctx)
        lm_logits = self.hug.lm_head(output[:, -1, :])
        return lm_logits

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
    def decoder_embedding(self,
                          src_ids: LongTensorT['B,SrcSeqLen'],
                          src_mask: IntTensorT['B,SrcSeqLen'],
                          ret_last: bool = True) -> \
            FloatTensorT['B,Opt[SrcSeqLen],EmbedLen']:
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.hug.model.forward(
            src_ids=src_ids, src_mask=src_mask)
        if ret_last:
            output = output[:, -1, :]
        output = self._change_output_if_configured(output)
        embedding = self.embedding_head(output)
        return embedding


    @overrides(DecoderTransformer)
    def decode_embedding_w_lm_logits(self,
                                     src_ids: LongTensorT['B,SrcSeqLen'],
                                     src_mask: IntTensorT['B,SrcSeqLen'],
                                     inference_context: InferenceContext = None) -> \
    FloatTensorT['B,2']:
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.hug.model.forward(
            src_ids=src_ids, src_mask=src_mask)
        output = output[:, -1, :]
        lm_logits = self.hug.lm_head(output)
        assert inference_context is not None, 'must provide inference context'
        no_yes_logits = inference_context.get_word_scores_from_logits(lm_logits)
        assert no_yes_logits.shape == (src_ids.shape[0],
                                       2)  # no_yes_score = nn.Tanh()(get_llama_no_yes_scores(lm_logits))

        output = output
        #         output = nn.Tanh()(output)
        assert self.embedding_head.weight.shape[1] == self.embed_dim + 2
        output = FloatTensorT(torch.cat([output, no_yes_score], dim=-1))
        embedding = self.embedding_head(output)
        return embedding

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
