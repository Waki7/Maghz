from __future__ import annotations

# from mgz.models.nlp.bart_interface import BARTHubInterface
import os

from mgz.version_control import ModelNode

os.putenv("PYTORCH_ENABLE_MPS_FALLBACK", "1")
from mgz.models.nlp.led import LEDForConditionalGeneration
import logging

'''
input_ids tensor([[    0, 31414,   232,  1437,     2,     2]])
attention_mask None
attn_weights torch.Size([16, 6, 6])
attn_probs torch.Size([16, 6, 6])
'''


def led_main_train():
    logging.basicConfig(level=logging.WARNING)
    batch_size = 4
    # Initializing a BART facebook/bart-large style configuration
    # model_name = "facebook/bart-base"
    # model_name = 'allenai/bart-large-multi_lexsum-long-tiny'
    # model_name = 'allenai/bart-large-multi_lexsum-long-short'
    model_name = 'allenai/led-base-16384-multi_lexsum-source-tiny'
    print('... loading model and tokenizer')
    ModelNode.load_from_id(LEDForConditionalGeneration, model_name, model_name)
    exit(3)


if __name__ == '__main__':
    led_main_train()
