from __future__ import annotations

import torch.utils.data
from transformers import PreTrainedTokenizerBase

import mgz.model_running.nlp_routines.model_routine_tagging as tagging
import mgz.settings as settings
from mgz.ds.sentence_datasets.gpt_input_augments import tag_question_augment, \
    SummarizePromptInput, PromptingInput, ContextPromptingInput, \
    PromptConfig
from mgz.ds.sentence_datasets.sentence_datasets import subsequent_mask, \
    strings_to_padded_id_tensor_w_mask, prompts_to_padded_id_tensor_w_mask
from mgz.models.nlp.base_transformer import BaseTransformer, \
    EncoderDecoderTransformer, DecoderTransformer, InferenceContext, ModelType
from mgz.typing import *


@torch.no_grad()
def embedding_controller_from_texts(model: BaseTransformer, texts: List[str],
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
        if model.MODEL_TYPE == ModelType.EncoderDecoderTransformer:
            max_src_len = model.get_max_encoder_positions()
        else:
            max_src_len = model.get_max_decoder_positions() - 1

    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(texts, tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)
    if model.MODEL_TYPE == ModelType.DecoderTransformer:
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.decoder_embedding(
            src_ids=src_ids,
            src_mask=src_mask,
        )
    elif model.MODEL_TYPE == ModelType.EncoderDecoderTransformer:
        tgt_ids, tgt_mask = strings_to_padded_id_tensor_w_mask(
            [tokenizer.sep_token], tokenizer,
            max_len=1,
            device=settings.DEVICE)
        embedding: FloatTensorT[
            'TaskSize,EmbedLen'] = model.encoder_decoder_embedding(
            src_ids=src_ids,
            tgt_ids=tgt_ids,
            src_mask=src_mask,
            tgt_mask=tgt_mask
        )
    else:
        raise NotImplementedError
    return embedding


def forward_controller_from_texts(model: BaseTransformer, texts: List[str],
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
def generate_controller_from_texts(model: DecoderTransformer, texts: List[str],
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
def _qa_controller_from_prompts(model: DecoderTransformer,
                                prompts: List[PromptingInput],
                                tokenizer: PreTrainedTokenizerBase,
                                max_src_len: int = None,
                                max_new_tokens=1000,
                                return_just_new: bool = True,
                                ) -> List[str]:
    """
    Generates sequences using a Transformer-based model for a list of input texts.

    Args:
        model (BaseTransformer): The Transformer-based model.
        prompts (List[str]): List of input texts to use as prompts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.

    Returns:
        Tensor: Generated sequences.
    """
    assert model.MODEL_TYPE == ModelType.DecoderTransformer
    if max_src_len is None:
        logging.info(
            f'max_src_len is None, setting to model max_decoder_positions - max_new_tokens {model.get_max_decoder_positions()} - {max_new_tokens}')
        max_src_len = model.get_max_decoder_positions() - max_new_tokens
    else:
        max_src_len = min(max_src_len,
                          model.get_max_decoder_positions() - max_new_tokens)

    src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(prompts,
                                                           tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)
    # for i in range(src_ids.shape[0]):
    #     print('src_ids', src_ids[i, :].shape)
    #     print('augmented_text: ', augmented_text[i])
    #     print('unique', torch.unique(src_ids[i, :]))
    generated_ids = model.generate(
        src_ids=src_ids,
        src_mask=src_mask,
        tgt_ids=src_ids,
        max_new_tokens=max_new_tokens)
    if return_just_new:
        generated_ids = generated_ids[:, src_ids.shape[-1]:]
    # if "7677625.1075844211594" in texts[0]:
    #     if tags:
    #         augmented_text = [augment_function(text, tag) for
    #                           text, tag in zip(texts[:1], tags[:1])]
    #     else:
    #         augmented_text = [augment_function(text) for
    #                           text in texts[:1]]
    #     print('INSIDE texts', augmented_text)
    #     src_ids, src_mask = strings_to_padded_id_tensor_w_mask(augmented_text,
    #                                                            tokenizer,
    #                                                            max_src_len,
    #                                                            settings.DEVICE)
    #     print('INSIDE src_ids', src_ids.shape)
    #     print('INSIDE src_mask', src_mask.shape)
    #     generated_ids = model.generate(
    #         src_ids=src_ids,
    #         src_mask=src_mask,
    #         tgt_ids=src_ids,
    #         max_new_tokens=max_new_tokens)
    #     print('INSIDE embedding',
    #           tokenizer.batch_decode(generated_ids[:, src_ids.shape[-1]:], skip_special_tokens=True))
    #     if return_just_new:
    #         generated_ids = generated_ids[:, src_ids.shape[-1]:]
    #
    #
    #
    #
    #     if tags:
    #         augmented_text = [augment_function(text, tag) for
    #                           text, tag in zip(texts, tags)]
    #     else:
    #         augmented_text = [augment_function(text) for
    #                           text in texts]
    #     print('INSIDE texts', augmented_text)
    #     src_ids, src_mask = strings_to_padded_id_tensor_w_mask(augmented_text,
    #                                                            tokenizer,
    #                                                            max_src_len,
    #                                                            settings.DEVICE)
    #     print('INSIDE src_ids', src_ids.shape)
    #     print('INSIDE src_mask', src_mask.shape)
    #     generated_ids = model.generate(
    #         src_ids=src_ids,
    #         src_mask=src_mask,
    #         tgt_ids=src_ids,
    #         max_new_tokens=max_new_tokens)
    #     print('INSIDE embedding',
    #           tokenizer.batch_decode(generated_ids[:, src_ids.shape[-1]:], skip_special_tokens=True))
    #     if return_just_new:
    #         generated_ids = generated_ids[:, src_ids.shape[-1]:]
    responses = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    return responses


@torch.no_grad()
def summarize_controller_from_texts(model: DecoderTransformer,
                                    texts: List[str],
                                    tokenizer: PreTrainedTokenizerBase,
                                    system_context: str = None,
                                    max_src_len: int = None,
                                    return_just_new: bool = True,
                                    word_limit: int = 50,
                                    prompt_config: PromptConfig = None,
                                    ) -> List[str]:
    if prompt_config is None:
        prompt_config = PromptConfig(system_context=system_context,
                                     model=model)
    prompts = SummarizePromptInput.from_list(prompt_config=prompt_config,
                                             document_texts=texts,
                                             word_limit=word_limit)
    with torch.cuda.amp.autocast(enabled=True):
        # TODO, confusion about the whole word limit vs token limit
        return _qa_controller_from_prompts(model, prompts, tokenizer,
                                           max_src_len=max_src_len,
                                           max_new_tokens=4 * word_limit,
                                           return_just_new=return_just_new, )


@torch.no_grad()
def summarize_controller_from_prompts(model: DecoderTransformer,
                                      prompts: List[SummarizePromptInput],
                                      tokenizer: PreTrainedTokenizerBase,
                                      max_src_len: int = None,
                                      return_just_new: bool = True,
                                      word_limit: int = None,
                                      ) -> List[str]:
    if word_limit is None:
        word_limit = prompts[0].word_limit
    with torch.cuda.amp.autocast(enabled=True):
        # TODO, confusion about the whole word limit vs token limit
        return _qa_controller_from_prompts(model, prompts, tokenizer,
                                           max_src_len=max_src_len,
                                           max_new_tokens=4 * word_limit,
                                           return_just_new=return_just_new, )


@torch.no_grad()
def tag_questions_controller_from_texts(model: DecoderTransformer,
                                        texts: List[str],
                                        tags: List[str],
                                        tokenizer: PreTrainedTokenizerBase,
                                        system_context: str = None,
                                        max_src_len: int = None,
                                        max_new_tokens=1000,
                                        return_just_new: bool = True,
                                        prompt_config: PromptConfig = None,
                                        ) -> List[str]:
    assert model.MODEL_TYPE == ModelType.DecoderTransformer
    assert len(texts) == len(tags)
    if prompt_config is None:
        prompt_config = PromptConfig(system_context=system_context,
                                     model=model)
    prompts = ContextPromptingInput.from_list(prompt_config=prompt_config,
                                              document_texts=texts,
                                              document_requests_list=tags)
    with torch.cuda.amp.autocast(enabled=True):
        return _qa_controller_from_prompts(model, prompts, tokenizer,
                                           tags=tags,
                                           max_src_len=max_src_len,
                                           max_new_tokens=max_new_tokens,
                                           return_just_new=return_just_new, )


@torch.no_grad()
def hybrid_generation_tagging_from_texts(model: DecoderTransformer,
                                         texts: List[str],
                                         tags: List[str],
                                         tokenizer: PreTrainedTokenizerBase,
                                         system_context: str = None,
                                         max_src_len: int = None,
                                         prompt_config: PromptConfig = None,
                                         ) -> \
        Tuple[FloatTensorT['B,EmbedLen'], ProbTensorT['2']]:
    """
    Computes embeddings for a list of input texts and corresponding tag texts
    using a Decoder Transformer.

       Args: model (DecoderTransformer): The Encoder-Decoder
       Transformer model. texts (List[str]): List of input texts to embed.
       tag_text (List[str]): List of tag texts. tokenizer (
       PreTrainedTokenizerBase): Tokenizer for text encoding. max_src_len (
       int, optional): Maximum source sequence length. Defaults to None.

       Returns:
           FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
       """
    assert model.MODEL_TYPE == ModelType.DecoderTransformer
    assert len(texts) == len(tags)
    if prompt_config is None:
        prompt_config = PromptConfig(system_context=system_context,
                                     model=model)
    prompts = ContextPromptingInput.from_list(prompt_config=prompt_config,
                                              document_texts=texts,
                                              document_requests_list=tags)

    if max_src_len is None:
        max_src_len = model.get_max_decoder_positions() - 1
    src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(prompts,
                                                           tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    # don't need tgt_mask because you are generating one token at a time
    embedding: FloatTensorT['B,EmbedLen']
    lm_logits: FloatTensorT['NClasses']
    embedding, lm_logits = model.decode_embedding_w_lm_logits(src_ids=src_ids,
                                                              src_mask=src_mask)
    no_yes_logits: FloatTensorT['2'] = InferenceContext(
        tokenizer).get_word_scores_from_logits(lm_logits)
    return embedding, ProbTensorT(torch.softmax(no_yes_logits, dim=-1))


@torch.no_grad()
def test(model: DecoderTransformer,
         texts: List[str],
         tag_text: List[str],
         tokenizer: PreTrainedTokenizerBase,
         max_src_len: int = None,
         ) -> \
        Tuple[FloatTensorT['B,EmbedLen'], ProbTensorT['2']]:
    """
    Computes embeddings for a list of input texts and corresponding tag texts
    using a Decoder Transformer.

       Args: model (DecoderTransformer): The Encoder-Decoder
       Transformer model. texts (List[str]): List of input texts to embed.
       tag_text (List[str]): List of tag texts. tokenizer (
       PreTrainedTokenizerBase): Tokenizer for text encoding. max_src_len (
       int, optional): Maximum source sequence length. Defaults to None.

       Returns:
           FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
       """
    assert model.MODEL_TYPE == ModelType.DecoderTransformer
    assert len(texts) == len(tag_text)
    if max_src_len is None:
        max_src_len = model.get_max_decoder_positions() - 1
        logging.info(f"max_src_len: {max_src_len}")
    src_qa_text = [tag_question_augment(text, tag) for text, tag in
                   zip(texts, tag_text)]
    logging.info("src_qa_text: %s", src_qa_text[0])
    logging.info("src_qa_text: %s", src_qa_text[1])
    src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(src_qa_text,
                                                           tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)
    inference_context = InferenceContext(tokenizer, model=model)
    # don't need tgt_mask because you are generating one token at a time
    embedding: FloatTensorT['B,EmbedLen']
    no_yes_logits: FloatTensorT['2']
    embedding, lm_logits = model.decode_relevance_at_triggers(src_ids=src_ids,
                                                              src_mask=src_mask,
                                                              inference_context=inference_context)

    assert inference_context is not None, 'must provide inference context'
    no_yes_logits = inference_context.get_word_scores_from_logits_at_triggers(
        src_ids, lm_logits)
    return embedding, no_yes_logits

    print('lm_logits ', lm_logits.shape)
    last_words = torch.argmax(lm_logits[:, -10:, :], dim=-1)
    decoded = tokenizer.batch_decode(last_words,
                                     skip_special_tokens=True)
    print('decoded ', decoded)

    # return embedding, ProbTensorT(torch.softmax(no_yes_logits, dim=-1))


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
                                 tags: List[str],
                                 tokenizer: PreTrainedTokenizerBase,
                                 max_src_len: int = None,
                                 max_tgt_len: int = None
                                 ) -> FloatTensorT['B,EmbedLen']:
    """
    Computes embeddings for a list of input texts and corresponding tag texts using an Encoder-Decoder Transformer.

    Args:
        model (EncoderDecoderTransformer): The Encoder-Decoder Transformer model.
        texts (List[str]): List of input texts to embed.
        tags (List[str]): List of tag texts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.
        max_src_len (int, optional): Maximum source sequence length. Defaults to None.
        max_tgt_len (int, optional): Maximum target sequence length. Defaults to None.

    Returns:
        FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
    """
    assert model.MODEL_TYPE == ModelType.EncoderDecoderTransformer
    if max_src_len is None:
        max_src_len = model.get_max_encoder_positions()
    if max_tgt_len is None:
        max_tgt_len = model.get_max_decoder_positions() - 1
    src_ids, src_mask = strings_to_padded_id_tensor_w_mask(texts, tokenizer,
                                                           max_src_len,
                                                           settings.DEVICE)

    tgt_ids, tgt_mask = strings_to_padded_id_tensor_w_mask(tags, tokenizer,
                                                           max_tgt_len,
                                                           settings.DEVICE)
    # don't need tgt_mask because you are generating one token at a time
    return model.encoder_decoder_embedding(src_ids=src_ids, tgt_ids=tgt_ids,
                                           src_mask=src_mask, tgt_mask=tgt_mask)


@torch.no_grad()
def prompt_lm_logits_controller(model: DecoderTransformer,
                                texts: List[str],
                                tags: List[str],
                                tokenizer: PreTrainedTokenizerBase,
                                system_context: str = None,
                                max_src_len: int = None,
                                prompt_config: PromptConfig = None) -> \
        FloatTensorT['B,EmbedLen']:
    """
    Computes embeddings for a list of input texts and corresponding tag texts using an Encoder-Decoder Transformer.

    Args:
        prompt_config:
        model (EncoderDecoderTransformer): The Encoder-Decoder Transformer model.
        texts (List[str]): List of input texts to embed.
        tags (List[str]): List of tag texts.
        tokenizer (PreTrainedTokenizerBase): Tokenizer for text encoding.
        max_tgt_len (int, optional): Maximum target sequence length. Defaults to None.

    Returns:
        FloatTensorT['B,EmbedLen']: Tensor containing the embeddings.
    """
    assert model.MODEL_TYPE == ModelType.DecoderTransformer
    assert len(texts) == len(tags)
    if prompt_config is None:
        prompt_config = PromptConfig(system_context=system_context,
                                     model=model)
    prompts = ContextPromptingInput.from_list(prompt_config=prompt_config,
                                              document_texts=texts,
                                              document_requests_list=tags)

    if max_src_len is None:
        max_src_len = model.get_max_decoder_positions() - 1
    src_ids, src_mask = prompts_to_padded_id_tensor_w_mask(
        prompts, tokenizer, max_src_len, settings.DEVICE)
    # don't need tgt_mask because you are generating one token at a time
    lm_logits = \
        model.decode_embedding_w_lm_logits(src_ids=src_ids, src_mask=src_mask)[
            1]
    return lm_logits


def main():
    pass


if __name__ == '__main__':
    main()
