from __future__ import annotations
from __future__ import annotations

import unittest

from transformers import LEDConfig

from mgz import settings
from mgz.model_running.nlp_routines.model_routine_tagging import \
    predict_with_centers
from mgz.model_running.run_ops import tagging_embedding_controller, \
    embedding_controller
from mgz.models.nlp.base_transformer import BaseTransformer
from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.models.nlp.mistral import MistralForCausalLM
from mgz.typing import *
from mgz.version_control import ModelNode


class TestBert(unittest.TestCase):
    def setUp(self):
        # CREATE A ENCODER DOCODER MODEL
        self.model_cache = {}
        self.tokenizer_cache = {}

    def load_model_if_needed(self, model_cls: Type[BaseTransformer],
                             model_name: str, quantize_8bit: bool = False):
        if model_name not in self.model_cache:
            model_node: ModelNode = \
                ModelNode.load_from_id(model_cls,
                                       model_name,
                                       model_name, quantize_8bit=quantize_8bit)
            model_node.model.eval()
            self.model_cache[model_name] = model_node.model
            self.tokenizer_cache[model_name] = model_node.tokenizer
            settings.print_gpu_usage()
        return self.model_cache[model_name], self.tokenizer_cache[model_name]

    def test_tagging_embedding_controller_shape(self):
        model: LEDForConditionalGeneration
        model, tokenizer = self.load_model_if_needed(
            LEDForConditionalGeneration,
            'allenai/led-base-16384-multi_lexsum-source-long')

        tags = ['hello', 'ground']
        with torch.no_grad():
            embedding: FloatTensorT[
                'TaskSize,EmbedLen'] = \
                tagging_embedding_controller(model,
                                             ['hello world', 'hello world'],
                                             tags, tokenizer)
        assert 768 == model.embed_dim
        self.assertEqual(embedding.shape, (len(tags), 768))

    def test_tagging_embedding(self):
        with ((torch.cuda.amp.autocast(enabled=True))):
            model: MistralForCausalLM
            model, tokenizer = self.load_model_if_needed(MistralForCausalLM,
                                                         'openchat/openchat_3.5',
                                                         quantize_8bit=True)
            config = cast(LEDConfig, model.config)
            tag_text = "Is this about cats?: "
            max_len = 4096
            with torch.no_grad():
                positive_examples = [tag_text + 'This is about cats',
                                     tag_text + 'This is about dogs']
                pos_embedding: FloatTensorT['EmbedLen'] = \
                    embedding_controller(model, positive_examples,
                                         tokenizer, max_src_len=max_len).mean(0,
                                                                              keepdim=False)
                settings.print_gpu_usage()

                negative_examples = [tag_text + 'This is about trains',
                                     tag_text + 'This is about cars']
                neg_embedding: FloatTensorT['EmbedLen'] = \
                    embedding_controller(model, negative_examples,
                                         tokenizer, max_src_len=max_len).mean(0,
                                                                              keepdim=False)

                support_embedding = FloatTensorT(
                    torch.stack([neg_embedding, pos_embedding], dim=0),
                    'NClasses,EmbedLen')

                del pos_embedding
                del neg_embedding
                settings.empty_cache()

                query_examples = [tag_text + 'Cats jump high',
                                  tag_text + 'Cars go fast']
                query_embedding: FloatTensorT[
                    'TaskSize,EmbedLen'] = \
                    embedding_controller(model, query_examples, tokenizer,
                                         max_src_len=max_len)

                probs = predict_with_centers(support_embedding, query_embedding)
                print(probs)
        self.assertEqual(probs.shape, (2, 2))
