from __future__ import annotations
from __future__ import annotations

import unittest

from transformers import LEDConfig

from mgz.model_running.run_ops import tagging_embedding_controller
from mgz.models.nlp.led import LEDForConditionalGeneration
from mgz.typing import *
from mgz.version_control import ModelNode


#
# MAKE THIS IN ORDER OF THE ACTUAL FLOW OF BERT

# import altair as alt
# import GPUtil

class TestBert(unittest.TestCase):
    def setUp(self):
        # CREATE A ENCODER DOCODER MODEL
        model_node: ModelNode = \
            ModelNode.load_from_id(LEDForConditionalGeneration,
                                   'allenai/led-base-16384-multi_lexsum-source-long',
                                   'allenai/led-base-16384-multi_lexsum-source-long')
        self.model = model_node.model
        self.config = cast(LEDConfig, self.model.config)
        self.tokenizer = model_node.tokenizer

        # CREATE A DECODER MODEL

    def test_tagging_embedding_controller_shape(self):
        tags = ['hello', 'ground']
        with torch.no_grad():
            embedding: FloatTensorT[
                'TaskSize,EmbedLen'] = \
                tagging_embedding_controller(self.model,
                                             ['hello world', 'hello world'],
                                             tags, self.tokenizer)
        assert 768 == self.config.d_model
        self.assertEqual(embedding.shape, (len(tags), 768))
