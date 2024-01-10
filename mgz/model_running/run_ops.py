from __future__ import annotations

import mgz.model_running.nlp_routines.model_routine_tagging as tagging
import mgz.settings as settings
import torch.utils.data
from mgz.ds.sentence_datasets.gpt_input_augments import summarization_augment, \
    tag_question_augment
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask, \
    strings_to_padded_id_tensor_w_mask
from mgz.models.nlp.base_transformer import BaseTransformer, \
    EncoderDecoderTransformer, DecoderTransformer
from mgz.typing import *
from transformers import PreTrainedTokenizerBase


@torch.no_grad()
def embedding_controller(model: BaseTransformer, texts: List[str],
                         tokenizer: PreTrainedTokenizerBase,
                         max_src_len: int = None,
                         ) -> FloatTensorT['B,EmbedLen']:
    """
    Computes embeddings for a list of input texts using a Transformer-based model.

    Args:
        model (BaseTransformer): The Transformer-based model.
        texts (List[str]): List of input texts to embed.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.
        max_src_len (int, optional): Maximum source sequence length. Defaults to None.

    Returns:
        FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
    """
    if max_src_len is None:
        if isinstance(model, EncoderDecoderTransformer):
            max_src_len = model.get_max_encoder_positions()
        else:
            max_src_len = model.get_max_decoder_positions()

    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(texts, tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    tgt_ids, tgt_mask = strings_to_padded_id_tensor_w_mask(
        [tokenizer.sep_token], tokenizer,
        max_len=1,
        device=settings.DEVICE)

    if isinstance(model, EncoderDecoderTransformer):
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
    elif isinstance(model, DecoderTransformer):
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.decoder_embedding(
            src_ids=src_ids,
            src_mask=src_mask,
        )
    else:
        raise NotImplementedError
    return embedding


def forward_controller(model: BaseTransformer, texts: List[str],
                       tokenizer: PreTrainedTokenizerBase):
    """
    Forward pass through a Transformer-based model for a list of input texts.

    Args:
        model (BaseTransformer): The Transformer-based model.
        texts (List[str]): List of input texts to process.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.

    Returns:
        Tensor: Output tensor from the model.
    """
    batch_encoding = tokenizer(texts, return_tensors="pt")
    src_ids = batch_encoding.input_ids.to(settings.DEVICE)
    batch_size = len(texts)
    tgt_ids = torch.LongTensor([tokenizer.sep_token_id]).unsqueeze(0).to(
        settings.DEVICE).repeat(batch_size, 1)
    src_mask = batch_encoding.attention_mask.to(settings.DEVICE)
    tgt_mask = (tgt_ids != tokenizer.pad_token_id).unsqueeze(-2)
    tgt_mask = tgt_mask & subsequent_mask(tgt_ids.size(-1)).type_as(
        tgt_mask.data
    )
    # don't need tgt_mask because you are generating one token at a time
    return model.forward(src_ids=src_ids, tgt_ids=tgt_ids,
                         src_mask=src_mask, tgt_mask=tgt_mask)


@torch.no_grad()
def generate_controller(model: DecoderTransformer, texts: List[str],
                        tokenizer: PreTrainedTokenizerBase,
                        ):
    """
    Generates sequences using a Transformer-based model for a list of input texts.

    Args:
        model (DecoderTransformer): The Transformer-based model.
        texts (List[str]): List of input texts to use as prompts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.

    Returns:
        Tensor: Generated sequences.
    """
    model_inputs = tokenizer(texts, return_tensors="pt").to(
        settings.DEVICE)
    # model.config.max_length = 7
    generated_ids = model.generate(
        src_ids=model_inputs.input_ids,
        src_mask=model_inputs.attention_mask,
        tgt_ids=model_inputs.input_ids,
        max_new_tokens=2000)
    return tokenizer.batch_decode(generated_ids, skip_special_tokens=True)


@torch.no_grad()
def summarize_controller(model: DecoderTransformer, texts: List[str],
                         tokenizer: PreTrainedTokenizerBase,
                         max_src_len: int = None,
                         max_new_tokens=1000,
                         return_just_new: bool = True,
                         ) -> List[str]:
    """
    Generates sequences using a Transformer-based model for a list of input texts.

    Args:
        model (BaseTransformer): The Transformer-based model.
        texts (List[str]): List of input texts to use as prompts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.

    Returns:
        Tensor: Generated sequences.
    """
    assert isinstance(model, DecoderTransformer)

    if max_src_len is None:
        max_src_len = model.get_max_decoder_positions() - max_new_tokens
    max_src_len = min(max_src_len,
                      model.get_max_decoder_positions() - max_new_tokens)

    texts = [summarization_augment(text) for text in texts]

    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(texts,
                                                           tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    # model.config.max_length = 7
    generated_ids = model.generate(
        src_ids=src_ids,
        src_mask=src_mask,
        tgt_ids=src_ids,
        max_new_tokens=max_new_tokens)

    if return_just_new:
        generated_ids = generated_ids[:, src_ids.shape[-1]:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


@torch.no_grad()
def hybrid_generation_tagging(model: DecoderTransformer,
                              texts: List[str],
                              tag_text: List[str],
                              tokenizer: PreTrainedTokenizerBase,
                              max_src_len: int = None,
                              ) -> \
        Tuple[FloatTensorT['B,EmbedLen'], ProbTensorT['2']]:
    """
    Computes embeddings for a list of input texts and corresponding tag texts
    using a Decoder Transformer.

       Args: model (EncoderDecoderTransformer): The Encoder-Decoder
       Transformer model. texts (List[str]): List of input texts to embed.
       tag_text (List[str]): List of tag texts. tokenizer (
       PreTrainedTokenizerBase): Tokenizer for text encoding. max_src_len (
       int, optional): Maximum source sequence length. Defaults to None.

       Returns:
           FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
       """
    assert isinstance(model, DecoderTransformer)
    assert len(texts) == len(tag_text)
    if max_src_len is None:
        max_src_len = model.get_max_decoder_positions()
    src_qa_text = [tag_question_augment(text, tag) for text, tag in
                   zip(texts, tag_text)]
    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(src_qa_text,
                                                           tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    # don't need tgt_mask because you are generating one token at a time
    embedding: FloatTensorT['B,EmbedLen']
    no_yes_logits: FloatTensorT['2']
    embedding, no_yes_logits = model.decode_relevance(src_ids=src_ids,
                                                      src_mask=src_mask)
    return embedding, ProbTensorT(torch.softmax(no_yes_logits, dim=-1))


@torch.no_grad()
def predict_probs_from_optional_centers(
        query_embedding: FloatTensorT['NQuery,EmbedLen'],
        no_yes_scores: ProbTensorT['NQuery,VocabSize'],
        neg_support_center: FloatTensorT['EmbedLen'] = None,
        pos_support_center: FloatTensorT['EmbedLen'] = None,
        n_supports: int = None
) -> ProbTensorT['NQuery,NClasses']:
    assert len(query_embedding.shape) == 2
    assert len(no_yes_scores.shape) == 2
    if neg_support_center is not None and pos_support_center is not None:
        support_embedding = FloatTensorT(
            torch.stack([neg_support_center, pos_support_center], dim=0),
            '2,EmbedLen')
    else:
        support_embedding = None
    return tagging.predict_probs_with_optional_prototypes(
        query_embedding=query_embedding,
        no_yes_probs=no_yes_scores,
        support_embedding=support_embedding,
        n_supports=n_supports)


@torch.no_grad()
def tagging_embedding_controller(model: EncoderDecoderTransformer,
                                 texts: List[str],
                                 tag_text: List[str],
                                 tokenizer: PreTrainedTokenizerBase,
                                 max_src_len: int = None,
                                 max_tgt_len: int = None
                                 ) -> FloatTensorT['B,EmbedLen']:
    """
    Computes embeddings for a list of input texts and corresponding tag texts using an Encoder-Decoder Transformer.

    Args:
        model (EncoderDecoderTransformer): The Encoder-Decoder Transformer model.
        texts (List[str]): List of input texts to embed.
        tag_text (List[str]): List of tag texts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.
        max_src_len (int, optional): Maximum source sequence length. Defaults to None.
        max_tgt_len (int, optional): Maximum target sequence length. Defaults to None.

    Returns:
        FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
    """
    assert isinstance(model, EncoderDecoderTransformer)
    if max_src_len is None:
        max_src_len = model.get_max_encoder_positions()
    if max_tgt_len is None:
        max_tgt_len = model.get_max_decoder_positions()

    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(texts, tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    tgt_ids, tgt_mask = strings_to_padded_id_tensor_w_mask(tag_text, tokenizer,
                                                           max_tgt_len,
                                                           settings.DEVICE)
    # don't need tgt_mask because you are generating one token at a time
    return model.encoder_decoder_embedding(src_ids=src_ids, tgt_ids=tgt_ids,
                                           src_mask=src_mask, tgt_mask=tgt_mask)


def main():
    pass


if __name__ == '__main__':
    main()
