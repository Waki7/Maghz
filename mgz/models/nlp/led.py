""" PyTorch LED model."""
from __future__ import annotations

import json
import math
import os
import random
from abc import ABC

import torch.utils.checkpoint
import transformers as hug
from torch import nn
from transformers import BitsAndBytesConfig
from transformers.activations import ACT2FN
from transformers.models.led.configuration_led import LEDConfig

import mgz.settings as settings
from mgz.models.nlp.base_transformer import BaseTransformer, TransformerContext, \
    quantize_model, EncoderDecoderTransformer
from mgz.models.nlp.utils_attention import _attention
from mgz.typing import *


class LEDLearnedPositionalEmbedding(nn.Embedding):
    """
    This module learns positional embeddings up to a fixed maximum size.
    """

    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__(num_embeddings, embedding_dim)

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

        return super().forward(positions)


# Copied from transformers.models.longformer.modeling_longformer.LongformerSelfAttention with Longformer->LEDEncoder
class LEDEncoderSelfAttention(nn.Module):

    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attention_window: int = 512,
                 dropout: ProbT = 0.0,
                 ):
        super().__init__()
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"The hidden size ({embed_dim}) is not a multiple of the number of attention "
                f"heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = int(embed_dim / num_heads)
        self.embed_dim = embed_dim

        self.query = nn.Linear(embed_dim, self.embed_dim)
        self.key = nn.Linear(embed_dim, self.embed_dim)
        self.value = nn.Linear(embed_dim, self.embed_dim)

        # separate projection layers for tokens with global attention
        self.query_global = nn.Linear(embed_dim, self.embed_dim)
        self.key_global = nn.Linear(embed_dim, self.embed_dim)
        self.value_global = nn.Linear(embed_dim, self.embed_dim)

        self.dropout = dropout

        attention_window = attention_window
        assert (
                attention_window % 2 == 0
        ), f"`attention_window` for layer {self.layer_id} has to be an even value. Given {attention_window}"
        assert (
                attention_window > 0
        ), f"`attention_window` for layer {self.layer_id} has to be positive. Given {attention_window}"

        self.one_sided_attn_window_size = attention_window // 2

    def forward(
            self,
            hidden_states,
            mask=None,
            is_index_masked: Optional[torch.Tensor] = None,
            is_index_global_attn: IntTensorT['B,TgtSeqLen'] = None,
            is_global_attn: bool = False,
    ):
        """
        [`LEDEncoderSelfAttention`] expects *len(hidden_states)* to be multiple of *attention_window*. Padding to
        *attention_window* happens in [`LEDEncoderModel.forward`] to avoid redoing the padding on each layer.

        The *attention_mask* is changed in [`LEDEncoderModel.forward`] from 0, 1, 2 to:

            - -10000: no attention
            - 0: local attention
            - +10000: global attention
        """
        hidden_states = hidden_states.transpose(0, 1)

        # project hidden states
        query_vectors = self.query(hidden_states)
        key_vectors = self.key(hidden_states)
        value_vectors = self.value(hidden_states)

        seq_len, batch_size, embed_dim = hidden_states.size()
        assert (
                embed_dim == self.embed_dim
        ), f"hidden_states should have embed_dim = {self.embed_dim}, but has {embed_dim}"

        # normalize query
        query_vectors /= math.sqrt(self.head_dim)

        query_vectors = query_vectors.view(seq_len, batch_size, self.num_heads,
                                           self.head_dim).transpose(0, 1)
        key_vectors = key_vectors.view(seq_len, batch_size, self.num_heads,
                                       self.head_dim).transpose(0, 1)

        attn_scores = self._sliding_chunks_query_key_matmul(
            query_vectors, key_vectors, self.one_sided_attn_window_size
        )
        # values to pad for attention probs
        remove_from_windowed_attention_mask = (mask == 0)[:, :, None,
                                              None]

        # cast to fp32/fp16 then replace 1's with -inf
        float_mask = remove_from_windowed_attention_mask.type_as(
            query_vectors).masked_fill(
            remove_from_windowed_attention_mask,
            torch.finfo(query_vectors.dtype).min
        )
        # diagonal mask with zeros everywhere and -inf inplace of padding
        diagonal_mask = self._sliding_chunks_query_key_matmul(
            float_mask.new_ones(size=float_mask.size()), float_mask,
            self.one_sided_attn_window_size
        )

        # pad local attention probs
        attn_scores += diagonal_mask

        assert list(attn_scores.size()) == [
            batch_size,
            seq_len,
            self.num_heads,
            self.one_sided_attn_window_size * 2 + 1,
        ], (
            f"local_attn_probs should be of size ({batch_size}, {seq_len}, {self.num_heads},"
            f" {self.one_sided_attn_window_size * 2 + 1}), but is of size {attn_scores.size()}"
        )

        # compute local attention probs from global attention keys and contact over window dim
        if is_global_attn:
            # compute global attn indices required through out forward fn
            (
                max_num_global_attn_indices,
                is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero,
            ) = self._get_global_attn_indices(is_index_global_attn)
            # calculate global attn probs from global key

            global_key_attn_scores = self._concat_with_global_key_attn_probs(
                query_vectors=query_vectors,
                key_vectors=key_vectors,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
            )
            # concat to local_attn_probs
            # (batch_size, seq_len, num_heads, extra attention count + 2*window+1)
            attn_scores = torch.cat((global_key_attn_scores, attn_scores),
                                    dim=-1)

            # free memory
            del global_key_attn_scores
        attn_probs = nn.functional.softmax(
            attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        # softmax sometimes inserts NaN if all positions are masked, replace them with 0
        attn_probs = torch.masked_fill(attn_probs,
                                       is_index_masked[:, :, None, None], 0.0)

        attn_probs = attn_probs.type_as(attn_scores)

        # free memory
        del attn_scores
        # apply dropout
        attn_probs = nn.functional.dropout(attn_probs, p=self.dropout,
                                           training=self.training)

        value_vectors = value_vectors.view(seq_len, batch_size, self.num_heads,
                                           self.head_dim).transpose(0, 1)

        # compute local attention output with global attention value and add
        if is_global_attn:
            # compute sum of global and local attn
            attn_output = self._compute_attn_output_with_global_indices(
                value_vectors=value_vectors,
                attn_probs=attn_probs,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
            )
        else:
            # compute local attn only
            attn_output = self._sliding_chunks_matmul_attn_probs_value(
                attn_probs, value_vectors, self.one_sided_attn_window_size
            )

        assert attn_output.size() == (
            batch_size, seq_len, self.num_heads,
            self.head_dim), "Unexpected size"
        attn_output = attn_output.transpose(0, 1).reshape(seq_len, batch_size,
                                                          embed_dim).contiguous()

        # compute value for global attention and overwrite to attention output
        # TODO: remove the redundant computation
        if is_global_attn:
            global_attn_output, global_attn_probs = self._compute_global_attn_output_from_hidden(
                hidden_states=hidden_states,
                max_num_global_attn_indices=max_num_global_attn_indices,
                is_local_index_global_attn_nonzero=is_local_index_global_attn_nonzero,
                is_index_global_attn_nonzero=is_index_global_attn_nonzero,
                is_local_index_no_global_attn_nonzero=is_local_index_no_global_attn_nonzero,
                is_index_masked=is_index_masked,
            )

            # get only non zero global attn output
            nonzero_global_attn_output = global_attn_output[
                                         is_local_index_global_attn_nonzero[0],
                                         :,
                                         is_local_index_global_attn_nonzero[1]
                                         ]

            # overwrite values with global attention
            attn_output[is_index_global_attn_nonzero[
                        ::-1]] = nonzero_global_attn_output.view(
                len(is_local_index_global_attn_nonzero[0]), -1
            )
            # The attention weights for tokens with global attention are
            # just filler values, they were never used to compute the output.
            # Fill with 0 now, the correct values are in 'global_attn_probs'.
            attn_probs[is_index_global_attn_nonzero] = 0

        # outputs = ()
        # return outputs + (global_attn_probs,) if (
        #         is_global_attn and output_attentions) else outputs
        return attn_output.transpose(0, 1),

    @staticmethod
    def _pad_and_transpose_last_two_dims(hidden_states_padded, padding):
        """pads rows and then flips rows and columns"""
        hidden_states_padded = nn.functional.pad(
            hidden_states_padded, padding
        )  # padding value is not important because it will be overwritten
        hidden_states_padded = hidden_states_padded.view(
            *hidden_states_padded.size()[:-2], hidden_states_padded.size(-1),
            hidden_states_padded.size(-2)
        )
        return hidden_states_padded

    @staticmethod
    def _pad_and_diagonalize(chunked_hidden_states):
        """
        shift every row 1 step right, converting columns into diagonals.

        Example:

        ```python
        chunked_hidden_states: [
            0.4983,
            2.6918,
            -0.0071,
            1.0492,
            -1.8348,
            0.7672,
            0.2986,
            0.0285,
            -0.7584,
            0.4206,
            -0.0405,
            0.1599,
            2.0514,
            -1.1600,
            0.5372,
            0.2629,
        ]
        window_overlap = num_rows = 4
        ```

                     (pad & diagonalize) => [ 0.4983, 2.6918, -0.0071, 1.0492, 0.0000, 0.0000, 0.0000
                       0.0000, -1.8348, 0.7672, 0.2986, 0.0285, 0.0000, 0.0000 0.0000, 0.0000, -0.7584, 0.4206,
                       -0.0405, 0.1599, 0.0000 0.0000, 0.0000, 0.0000, 2.0514, -1.1600, 0.5372, 0.2629 ]
        """
        total_num_heads, num_chunks, window_overlap, hidden_dim = chunked_hidden_states.size()
        chunked_hidden_states = nn.functional.pad(
            chunked_hidden_states, (0, window_overlap + 1)
        )  # total_num_heads x num_chunks x window_overlap x (hidden_dim+window_overlap+1). Padding value is not important because it'll be overwritten
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, -1
        )  # total_num_heads x num_chunks x window_overlap*window_overlap+window_overlap
        chunked_hidden_states = chunked_hidden_states[
                                :, :, :-window_overlap
                                ]  # total_num_heads x num_chunks x window_overlap*window_overlap
        chunked_hidden_states = chunked_hidden_states.view(
            total_num_heads, num_chunks, window_overlap,
            window_overlap + hidden_dim
        )
        chunked_hidden_states = chunked_hidden_states[:, :, :, :-1]
        return chunked_hidden_states

    @staticmethod
    def _chunk(hidden_states, window_overlap):
        """convert into overlapping chunks. Chunk size = 2w, overlap size = w"""
        # non-overlapping chunks of size = 2w
        hidden_states = hidden_states.view(
            hidden_states.size(0),
            torch.div(hidden_states.size(1), (window_overlap * 2),
                      rounding_mode="trunc"),
            window_overlap * 2,
            hidden_states.size(2),
        )
        # use `as_strided` to make the chunks overlap with an overlap size = window_overlap
        chunk_size = list(hidden_states.size())
        chunk_size[1] = chunk_size[1] * 2 - 1

        chunk_stride = list(hidden_states.stride())
        chunk_stride[1] = chunk_stride[1] // 2
        return hidden_states.as_strided(size=chunk_size,
                                        stride=chunk_stride)

    @staticmethod
    def _mask_invalid_locations(input_tensor, affected_seq_len) -> torch.Tensor:
        beginning_mask_2d = input_tensor.new_ones(affected_seq_len,
                                                  affected_seq_len + 1).tril().flip(
            dims=[0])
        beginning_mask = beginning_mask_2d[None, :, None, :]
        ending_mask = beginning_mask.flip(dims=(1, 3))
        beginning_input = input_tensor[:, :affected_seq_len, :,
                          : affected_seq_len + 1]
        beginning_mask = beginning_mask.expand(beginning_input.size())
        input_tensor[:, :affected_seq_len, :,
        : affected_seq_len + 1] = torch.full_like(
            beginning_input, -float("inf")
        ).where(beginning_mask.bool(), beginning_input)
        ending_input = input_tensor[:, -affected_seq_len:, :,
                       -(affected_seq_len + 1):]
        ending_mask = ending_mask.expand(ending_input.size())
        input_tensor[:, -affected_seq_len:, :,
        -(affected_seq_len + 1):] = torch.full_like(
            ending_input, -float("inf")
        ).where(ending_mask.bool(), ending_input)

    def _sliding_chunks_query_key_matmul(self, query: torch.Tensor,
                                         key: torch.Tensor,
                                         window_overlap: int):
        """
        Matrix multiplication of query and key tensors using with a sliding window attention pattern. This
        implementation splits the input into overlapping chunks of size 2w (e.g. 512 for pretrained LEDEncoder) with an
        overlap of size window_overlap
        """
        batch_size, seq_len, num_heads, head_dim = query.size()
        assert (
                seq_len % (window_overlap * 2) == 0
        ), f"Sequence length should be multiple of {window_overlap * 2}. Given {seq_len}"
        assert query.size() == key.size()

        chunks_count = torch.div(seq_len, window_overlap,
                                 rounding_mode="trunc") - 1

        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size window_overlap * 2
        query = query.transpose(1, 2).reshape(batch_size * num_heads, seq_len,
                                              head_dim)
        key = key.transpose(1, 2).reshape(batch_size * num_heads, seq_len,
                                          head_dim)

        query = self._chunk(query, window_overlap)
        key = self._chunk(key, window_overlap)

        # matrix multiplication
        # bcxd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcyd: batch_size * num_heads x chunks x 2window_overlap x head_dim
        # bcxy: batch_size * num_heads x chunks x 2window_overlap x 2window_overlap
        diagonal_chunked_attention_scores = torch.einsum("bcxd,bcyd->bcxy", (
            query, key))  # multiply

        # convert diagonals into columns
        diagonal_chunked_attention_scores = self._pad_and_transpose_last_two_dims(
            diagonal_chunked_attention_scores, padding=(0, 0, 0, 1)
        )

        # allocate space for the overall attention matrix where the chunks are combined. The last dimension
        # has (window_overlap * 2 + 1) columns. The first (window_overlap) columns are the window_overlap lower triangles (attention from a word to
        # window_overlap previous words). The following column is attention score from each word to itself, then
        # followed by window_overlap columns for the upper triangle.

        diagonal_attention_scores = diagonal_chunked_attention_scores.new_zeros(
            (batch_size * num_heads, chunks_count + 1, window_overlap,
             window_overlap * 2 + 1)
        )

        # copy parts from diagonal_chunked_attention_scores into the combined matrix of attentions
        # - copying the main diagonal and the upper triangle
        diagonal_attention_scores[:, :-1, :,
        window_overlap:] = diagonal_chunked_attention_scores[
                           :, :, :window_overlap, : window_overlap + 1
                           ]
        diagonal_attention_scores[:, -1, :,
        window_overlap:] = diagonal_chunked_attention_scores[
                           :, -1, window_overlap:, : window_overlap + 1
                           ]
        # - copying the lower triangle
        diagonal_attention_scores[:, 1:, :,
        :window_overlap] = diagonal_chunked_attention_scores[
                           :, :, -(window_overlap + 1): -1, window_overlap + 1:
                           ]

        diagonal_attention_scores[:, 0, 1:window_overlap,
        1:window_overlap] = diagonal_chunked_attention_scores[
                            :, 0, : window_overlap - 1, 1 - window_overlap:
                            ]

        # separate batch_size and num_heads dimensions again
        diagonal_attention_scores = diagonal_attention_scores.view(
            batch_size, num_heads, seq_len, 2 * window_overlap + 1
        ).transpose(2, 1)

        self._mask_invalid_locations(diagonal_attention_scores, window_overlap)
        return diagonal_attention_scores

    def _sliding_chunks_matmul_attn_probs_value(
            self, attn_probs: torch.Tensor, value: torch.Tensor,
            window_overlap: int
    ):
        """
        Same as _sliding_chunks_query_key_matmul but for attn_probs and value tensors. Returned tensor will be of the
        same shape as `attn_probs`
        """
        batch_size, seq_len, num_heads, head_dim = value.size()

        assert seq_len % (window_overlap * 2) == 0
        assert attn_probs.size()[:3] == value.size()[:3]
        assert attn_probs.size(3) == 2 * window_overlap + 1
        chunks_count = torch.div(seq_len, window_overlap,
                                 rounding_mode="trunc") - 1
        # group batch_size and num_heads dimensions into one, then chunk seq_len into chunks of size 2 window overlap

        chunked_attn_probs = attn_probs.transpose(1, 2).reshape(
            batch_size * num_heads,
            torch.div(seq_len, window_overlap, rounding_mode="trunc"),
            window_overlap,
            2 * window_overlap + 1,
        )

        # group batch_size and num_heads dimensions into one
        value = value.transpose(1, 2).reshape(batch_size * num_heads, seq_len,
                                              head_dim)

        # pad seq_len with w at the beginning of the sequence and another window overlap at the end
        padded_value = nn.functional.pad(value,
                                         (0, 0, window_overlap, window_overlap),
                                         value=-1)

        # chunk padded_value into chunks of size 3 window overlap and an overlap of size window overlap
        chunked_value_size = (
            batch_size * num_heads, chunks_count + 1, 3 * window_overlap,
            head_dim)
        chunked_value_stride = padded_value.stride()
        chunked_value_stride = (
            chunked_value_stride[0],
            window_overlap * chunked_value_stride[1],
            chunked_value_stride[1],
            chunked_value_stride[2],
        )
        chunked_value = padded_value.as_strided(size=chunked_value_size,
                                                stride=chunked_value_stride)

        chunked_attn_probs = self._pad_and_diagonalize(chunked_attn_probs)

        context = torch.einsum("bcwd,bcdh->bcwh",
                               (chunked_attn_probs, chunked_value))
        return context.view(batch_size, num_heads, seq_len, head_dim).transpose(
            1, 2)

    @staticmethod
    def _get_global_attn_indices(is_index_global_attn):
        """compute global attn indices required throughout forward pass"""
        # helper variable
        num_global_attn_indices = is_index_global_attn.long().sum(dim=1)

        # max number of global attn indices in batch
        max_num_global_attn_indices = num_global_attn_indices.max()

        # indices of global attn
        is_index_global_attn_nonzero = is_index_global_attn.nonzero(
            as_tuple=True)

        # helper variable
        is_local_index_global_attn = torch.arange(
            max_num_global_attn_indices, device=is_index_global_attn.device
        ) < num_global_attn_indices.unsqueeze(dim=-1)

        # location of the non-padding values within global attention indices
        is_local_index_global_attn_nonzero = is_local_index_global_attn.nonzero(
            as_tuple=True)

        # location of the padding values within global attention indices
        is_local_index_no_global_attn_nonzero = (
                is_local_index_global_attn == 0).nonzero(as_tuple=True)
        return (
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
        )

    def _concat_with_global_key_attn_probs(
            self,
            key_vectors,
            query_vectors,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
    ):
        batch_size = key_vectors.shape[0]

        # create only global key vectors
        key_vectors_only_global = key_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads,
            self.head_dim
        )

        key_vectors_only_global[is_local_index_global_attn_nonzero] = \
            key_vectors[is_index_global_attn_nonzero]

        # (batch_size, seq_len, num_heads, max_num_global_attn_indices)
        attn_probs_from_global_key = torch.einsum("blhd,bshd->blhs", (
            query_vectors, key_vectors_only_global))

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)
        attn_probs_from_global_key[
        is_local_index_no_global_attn_nonzero[0],
        is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(attn_probs_from_global_key.dtype).min
        attn_probs_from_global_key = attn_probs_from_global_key.transpose(1, 3)

        return attn_probs_from_global_key

    def _compute_attn_output_with_global_indices(
            self,
            value_vectors,
            attn_probs,
            max_num_global_attn_indices,
            is_index_global_attn_nonzero,
            is_local_index_global_attn_nonzero,
    ):
        batch_size = attn_probs.shape[0]

        # cut local attn probs to global only
        attn_probs_only_global = attn_probs.narrow(-1, 0,
                                                   max_num_global_attn_indices)
        # get value vectors for global only
        value_vectors_only_global = value_vectors.new_zeros(
            batch_size, max_num_global_attn_indices, self.num_heads,
            self.head_dim
        )
        value_vectors_only_global[is_local_index_global_attn_nonzero] = \
            value_vectors[is_index_global_attn_nonzero]

        # use `matmul` because `einsum` crashes sometimes with fp16
        # attn = torch.einsum('blhs,bshd->blhd', (selected_attn_probs, selected_v))
        # compute attn output only global
        attn_output_only_global = torch.matmul(
            attn_probs_only_global.transpose(1, 2).clone(),
            value_vectors_only_global.transpose(1, 2).clone()
        ).transpose(1, 2)

        # reshape attn probs
        attn_probs_without_global = attn_probs.narrow(
            -1, max_num_global_attn_indices,
            attn_probs.size(-1) - max_num_global_attn_indices
        ).contiguous()

        # compute attn output with global
        attn_output_without_global = self._sliding_chunks_matmul_attn_probs_value(
            attn_probs_without_global, value_vectors,
            self.one_sided_attn_window_size
        )
        return attn_output_only_global + attn_output_without_global

    def _compute_global_attn_output_from_hidden(
            self,
            hidden_states,
            max_num_global_attn_indices,
            is_local_index_global_attn_nonzero,
            is_index_global_attn_nonzero,
            is_local_index_no_global_attn_nonzero,
            is_index_masked,
    ):
        seq_len, batch_size = hidden_states.shape[:2]

        # prepare global hidden states
        global_attn_hidden_states = hidden_states.new_zeros(
            max_num_global_attn_indices, batch_size, self.embed_dim)
        global_attn_hidden_states[is_local_index_global_attn_nonzero[::-1]] = \
            hidden_states[
                is_index_global_attn_nonzero[::-1]
            ]

        # global key, query, value
        global_query_vectors_only_global = self.query_global(
            global_attn_hidden_states)
        global_key_vectors = self.key_global(hidden_states)
        global_value_vectors = self.value_global(hidden_states)

        # normalize
        global_query_vectors_only_global /= math.sqrt(self.head_dim)

        # reshape
        global_query_vectors_only_global = (
            global_query_vectors_only_global.contiguous()
            .view(max_num_global_attn_indices, batch_size * self.num_heads,
                  self.head_dim)
            .transpose(0, 1)
        )  # (batch_size * self.num_heads, max_num_global_attn_indices, head_dim)
        global_key_vectors = (
            global_key_vectors.contiguous().view(-1,
                                                 batch_size * self.num_heads,
                                                 self.head_dim).transpose(0, 1)
        )  # batch_size * self.num_heads, seq_len, head_dim)
        global_value_vectors = (
            global_value_vectors.contiguous().view(-1,
                                                   batch_size * self.num_heads,
                                                   self.head_dim).transpose(0,
                                                                            1)
        )  # batch_size * self.num_heads, seq_len, head_dim)

        # compute attn scores
        global_attn_scores = torch.bmm(global_query_vectors_only_global,
                                       global_key_vectors.transpose(1, 2))

        assert list(global_attn_scores.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            seq_len,
        ], (
            "global_attn_scores have the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, seq_len)}, but is"
            f" {global_attn_scores.size()}."
        )

        global_attn_scores = global_attn_scores.view(batch_size, self.num_heads,
                                                     max_num_global_attn_indices,
                                                     seq_len)

        # need to transpose since ONNX export only supports consecutive indexing: https://pytorch.org/docs/stable/onnx.html#writes-sets
        global_attn_scores = global_attn_scores.transpose(1, 2)
        global_attn_scores[
        is_local_index_no_global_attn_nonzero[0],
        is_local_index_no_global_attn_nonzero[1], :, :
        ] = torch.finfo(global_attn_scores.dtype).min
        global_attn_scores = global_attn_scores.transpose(1, 2)

        global_attn_scores = global_attn_scores.masked_fill(
            is_index_masked[:, None, None, :],
            torch.finfo(global_attn_scores.dtype).min,
        )

        global_attn_scores = global_attn_scores.view(
            batch_size * self.num_heads, max_num_global_attn_indices, seq_len)

        # compute global attn probs
        global_attn_probs_float = nn.functional.softmax(
            global_attn_scores, dim=-1, dtype=torch.float32
        )  # use fp32 for numerical stability

        global_attn_probs = nn.functional.dropout(
            global_attn_probs_float.type_as(global_attn_scores), p=self.dropout,
            training=self.training
        )

        # global attn output
        global_attn_output = torch.bmm(global_attn_probs, global_value_vectors)

        assert list(global_attn_output.size()) == [
            batch_size * self.num_heads,
            max_num_global_attn_indices,
            self.head_dim,
        ], (
            "global_attn_output tensor has the wrong size. Size should be"
            f" {(batch_size * self.num_heads, max_num_global_attn_indices, self.head_dim)}, but is"
            f" {global_attn_output.size()}."
        )

        global_attn_probs = global_attn_probs.view(batch_size, self.num_heads,
                                                   max_num_global_attn_indices,
                                                   seq_len)
        global_attn_output = global_attn_output.view(
            batch_size, self.num_heads, max_num_global_attn_indices,
            self.head_dim
        )
        return global_attn_output, global_attn_probs


class LEDEncoderAttention(nn.Module):
    def __init__(self,
                 embed_dim: int,
                 num_heads: int,
                 attention_window: int = 512,
                 dropout: ProbT = 0.0):
        super().__init__()
        self.longformer_self_attn = LEDEncoderSelfAttention(embed_dim=embed_dim,
                                                            num_heads=num_heads,
                                                            attention_window=attention_window,
                                                            dropout=dropout)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(
            self,
            hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            mask: IntTensorT['B,SrcSeqLen'] = None,
            is_index_masked: Optional[torch.Tensor] = None,
            is_index_global_attn: IntTensorT['B,TgtSeqLen'] = None,
            is_global_attn: bool = False,
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        """Input shape: Batch x Time x Channel"""

        self_outputs = self.longformer_self_attn(
            hidden_states=hidden_states,
            mask=mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
        )

        attn_output = self.output(self_outputs[0])
        return attn_output


class MultiHeadedAttention(nn.Module):
    """Multi-headed attention from 'Attention Is All You Need' paper"""

    def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            decoder_attention: bool = False,
            dropout: ProbT = 0.0,
            bias: bool = True,
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
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.out_proj = nn.Linear(embed_dim, embed_dim, bias=bias)

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
        if (self.decoder_attention and transformer_ctx is not None
                and transformer_ctx.in_generation):
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


class LEDEncoderLayer(nn.Module):
    def __init__(self, config: LEDConfig, layer_num: int):
        super().__init__()
        self.embed_dim = config.d_model
        self.self_attn = LEDEncoderAttention(
            embed_dim=self.embed_dim,
            num_heads=config.encoder_attention_heads,
            dropout=config.attention_dropout if self.training else None,
            attention_window=config.attention_window[layer_num],
        )
        self.self_attn_layer_norm = nn.LayerNorm(self.embed_dim)
        self.dropout = config.dropout
        self.activation_fn = ACT2FN[config.activation_function]
        self.activation_dropout = config.activation_dropout
        self.fc1 = nn.Linear(self.embed_dim, config.encoder_ffn_dim)
        self.fc2 = nn.Linear(config.encoder_ffn_dim, self.embed_dim)
        self.final_layer_norm = nn.LayerNorm(self.embed_dim)

    DONE = 0

    def forward(
            self,
            hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'],
            src_mask: IntTensorT['B,SrcSeqLen'],
            is_index_masked: IntTensorT['B,SrcSeqLen'],
            is_index_global_attn: IntTensorT['B,SrcSeqLen'],
            is_global_attn: bool = False
    ) -> FloatTensorT['B,SrcSeqLen,EmbedLen']:
        residual = hidden_states
        hidden_states = self.self_attn.forward(
            hidden_states=hidden_states,
            mask=src_mask,
            is_index_masked=is_index_masked,
            is_index_global_attn=is_index_global_attn,
            is_global_attn=is_global_attn,
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


class LEDDecoderLayer(nn.Module):
    def __init__(self, config: LEDConfig):
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
            tgt_mask: IntTensorT['B,SrcSeqLen,SrcSeqLen'],
            transformer_ctx: TransformerContext = None
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


class LEDPretrainedModel(EncoderDecoderTransformer, ABC):

    @classmethod
    def modules_to_apply_lora(cls) -> List[str]:
        return ['key', 'value', 'query']

    @classmethod
    def modules_to_not_convert(cls):
        return ['model.encoder.embed_tokens', 'model.decoder.embed_tokens',
                'lm_head']

    @classmethod
    def load_tokenizer(cls, tokenizer_id: str) -> Optional[
        hug.LEDTokenizerFast]:
        try:
            return hug.LEDTokenizerFast.from_pretrained(tokenizer_id)
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

    # For LED we need to pad the input ids and attention mask to be multiple of the attention window
    def _pre_encode_pad_if_needed(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            src_mask: IntTensorT['B,SrcSeqLen'],
            pad_token_id: int,
    ):
        seq_len = src_ids.shape[1]
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )
        if seq_len % attention_window == 0:
            return src_ids, src_mask
        """A helper function to pad tokens and mask to work with implementation of Longformer self-attention."""
        # padding
        attention_window = (
            self.config.attention_window
            if isinstance(self.config.attention_window, int)
            else max(self.config.attention_window)
        )

        if attention_window % 2 != 0:
            raise ValueError(
                f"`attention_window` should be an even value. Given {attention_window}")
        input_shape = src_ids.shape
        batch_size, seq_len = input_shape[:2]

        padding_len = (
                              attention_window - seq_len % attention_window) % attention_window
        if padding_len > 0:

            if src_ids is not None:
                src_ids = nn.functional.pad(src_ids, (0, padding_len),
                                            value=pad_token_id)

            src_mask = nn.functional.pad(
                src_mask, (0, padding_len), value=False
            )  # no attention on the padding tokens

        return src_ids, src_mask


class LEDEncoder(nn.Module):
    """
    Transformer encoder consisting of *config.encoder_layers* self attention layers. Each layer is a
    [`LEDEncoderLayer`].

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig,
                 embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()

        self.dropout = config.dropout
        self.layerdrop = config.encoder_layerdrop

        embed_dim = config.d_model
        self.padding_idx = config.pad_token_id
        self.max_source_positions = config.max_encoder_position_embeddings
        self.embed_scale = 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, embed_dim,
                                         self.padding_idx)
        self.config = config

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_source_positions,
            embed_dim,
        )
        self.layers = nn.ModuleList(
            [LEDEncoderLayer(config, layer_num=i) for i in
             range(config.encoder_layers)])
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

        # get masking tensors
        is_index_masked = src_mask == 0
        # todo, this never seems to be true, so we can probably remove it, understand why it was here
        is_index_global_attn = src_mask > 1
        is_global_attn = is_index_global_attn.flatten().any().item()

        encoder_layer: LEDEncoderLayer
        for idx, encoder_layer in enumerate(self.layers):
            # add LayerDrop (https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (
                    dropout_probability < self.layerdrop):  # skip the layer
                layer_outputs = None
            else:
                layer_outputs = encoder_layer.forward(
                    hidden_states=hidden_states,
                    src_mask=src_mask,
                    is_index_masked=is_index_masked,
                    is_index_global_attn=is_index_global_attn,
                    is_global_attn=is_global_attn
                )

            hidden_states = layer_outputs
        return hidden_states


class LEDDecoder(nn.Module):
    """
    Transformer decoder consisting of *config.decoder_layers* layers. Each layer is a [`LEDDecoderLayer`]

    Args:
        config: LEDConfig
        embed_tokens (nn.Embedding): output embedding
    """

    def __init__(self, config: LEDConfig,
                 embed_tokens: Optional[nn.Embedding] = None):
        super().__init__()
        self.dropout = config.dropout
        self.layerdrop = config.decoder_layerdrop
        self.padding_idx = config.pad_token_id
        self.max_target_positions = config.max_decoder_position_embeddings
        self.embed_scale = 1.0

        self.embed_tokens = nn.Embedding(config.vocab_size, config.d_model,
                                         self.padding_idx)

        if embed_tokens is not None:
            self.embed_tokens.weight = embed_tokens.weight

        self.embed_positions = LEDLearnedPositionalEmbedding(
            self.max_target_positions,
            config.d_model,
        )
        self.layers = nn.ModuleList(
            [LEDDecoderLayer(config) for _ in range(config.decoder_layers)])
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
        decoder_layer: LEDDecoderLayer
        for idx, decoder_layer in enumerate(self.layers):
            # add LayerDrop (https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if self.training and (dropout_probability < self.layerdrop):
                continue
            hidden_states = decoder_layer.forward(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                src_mask=src_mask,
                tgt_mask=tgt_mask,
                transformer_ctx=transformer_ctx.for_layer(
                    idx) if transformer_ctx else None
            )
        return hidden_states


class LEDModel(LEDPretrainedModel):
    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.config: LEDConfig = config
        padding_idx, vocab_size = config.pad_token_id, config.vocab_size
        self.shared = nn.Embedding(vocab_size, config.d_model, padding_idx)

        self.encoder = LEDEncoder(config, self.shared)
        self.decoder = LEDDecoder(config, self.shared)
        self.apply(self._init_weights)

    def get_input_embeddings(self):
        return self.shared

    def set_input_embeddings(self, value):
        self.shared = value
        self.encoder.embed_tokens = self.shared
        self.decoder.embed_tokens = self.shared

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen']) -> FloatTensorT[
        'B,SrcSeqLen,EmbedLen']:
        src_ids, src_mask = self._pre_encode_pad_if_needed(
            src_ids=src_ids,
            src_mask=src_mask,
            pad_token_id=self.config.pad_token_id,
        )
        return self.encoder.forward(src_ids=src_ids, src_mask=src_mask)

    def forward(
            self,
            src_ids: LongTensorT['B,SrcSeqLen'],
            tgt_ids: LongTensorT['B,TgtSeqLen'],
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen'] = None
    ) -> FloatTensorT['B,TgtSeqLen,EmbedLen']:
        src_ids, src_mask = self._pre_encode_pad_if_needed(
            src_ids=src_ids,
            src_mask=src_mask,
            pad_token_id=self.config.pad_token_id,
        )
        encoder_outputs = self.encoder.forward(src_ids=src_ids,
                                               src_mask=src_mask)
        decoder_hidden_state = self.decoder.forward(
            encoder_hidden_states=encoder_outputs,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask,
        )
        return decoder_hidden_state


class LEDForConditionalGeneration(LEDPretrainedModel):
    def get_encoder(self):
        return self.led.encoder

    def get_max_encoder_positions(self):
        return self.get_encoder().embed_positions.weight.shape[0]

    def get_decoder(self):
        return self.led.decoder

    def get_max_decoder_positions(self):
        return self.get_decoder().embed_positions.weight.shape[0]

    @classmethod
    def load_model(cls, path: str,
                   quantization_config: BitsAndBytesConfig = None) -> Optional[
        LEDForConditionalGeneration]:
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
            model = LEDForConditionalGeneration(LEDConfig.from_dict(config))
            if quantization_config and quantization_config.load_in_8bit:
                model = quantize_model(model)
            model.load_state_dict(torch.load(os.path.join(path, 'weights.bin'),
                                             map_location=torch.device('cpu')))
            model.to(settings.DEVICE)
            return model
        except FileNotFoundError:
            return None

    @classmethod
    def initial_save(cls, model_id: str, path: str):
        if not os.path.exists(path):
            os.makedirs(path)
        model_hug = hug.LEDForConditionalGeneration.from_pretrained(
            model_id).to(settings.DEVICE)
        with open(os.path.normpath(os.path.join(path, 'config.json')),
                  'w') as f:
            json.dump(model_hug.config.to_dict(), f)
        torch.save(model_hug.state_dict(),
                   os.path.normpath(os.path.join(path, 'weights.bin')))
        tokenizer = hug.LEDTokenizerFast.from_pretrained(model_id)
        tokenizer.save_pretrained(path)

    def __init__(self, config: LEDConfig):
        super().__init__(config)
        self.led = LEDModel(config)
        self.register_buffer("final_logits_bias",
                             torch.zeros((1, self.led.shared.num_embeddings)))
        self.lm_head = nn.Linear(config.d_model,
                                 self.led.shared.num_embeddings, bias=False)

        # Initialize weights and apply final processing
        self.apply(self._init_weights)

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

    def encode(self, src_ids: LongTensorT['B,SrcSeqLen'],
               src_mask: IntTensorT['B,SrcSeqLen']):
        src_ids, src_mask = self._pre_encode_pad_if_needed(
            src_ids=src_ids,
            src_mask=src_mask,
            pad_token_id=self.config.pad_token_id,
        )
        return self.led.encode(src_ids=src_ids, src_mask=src_mask)

    def decode(self,
               transformer_ctx: TransformerContext,
               generation_ids: LongTensorT['B,1'],
               src_mask: IntTensorT['B,SrcSeqLen'] = None) -> \
            LogitsTensorT['B,1,VocabSize']:
        output = self.led.decoder.forward(
            encoder_hidden_states=transformer_ctx.encoder_memory,
            tgt_ids=generation_ids,
            src_mask=src_mask, transformer_ctx=transformer_ctx
        )

        lm_logits = self.lm_head(output)
        lm_logits = lm_logits + self.final_logits_bias.to(lm_logits.device)
        return lm_logits

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
        output: FloatTensorT['B,TgtSeqLen,EmbedLen'] = self.led.forward(
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
            src_mask: IntTensorT['B,1|TgtSeqLen,SrcSeqLen'],
            tgt_mask: IntTensorT['B,TgtSeqLen,TgtSeqLen']
    ) -> FloatTensorT['B,TgtSeqLen,OutNClasses']:
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


class LEDClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

    def __init__(
            self,
            input_dim: int,
            inner_dim: int,
            num_classes: int,
            pooler_dropout: float,
    ):
        super().__init__()
        self.dense = nn.Linear(input_dim, inner_dim)
        self.dropout = nn.Dropout(p=pooler_dropout)
        self.out_proj = nn.Linear(inner_dim, num_classes)

    def forward(self, hidden_states: torch.Tensor):
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.dense(hidden_states)
        hidden_states = torch.tanh(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.out_proj(hidden_states)
        return hidden_states
