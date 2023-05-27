import torch
from transformers import BartTokenizer

import settings
from mgz.models.nlp.bart import BartForConditionalGeneration
from mgz.run_ops.run_ops import generate_controller, forward_controller
from mgz.typing import *
import mgz.models.nlp.bart_orig as hug

use_mgz = True
use_hug = True
use_generation = True
use_encode = False

text = 'startval'
# model_name = 'facebook/bart-large'
model_name = 'facebook/bart-large-xsum'  # has a bug where it doesn't have 'mask' in its embedding table, or something like that
# model_name = 'facebook/bart-base'

tokenizer = BartTokenizer.from_pretrained(model_name)
print(tokenizer.special_tokens_map)
model_mgz: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
    model_name).to(settings.DEVICE)
print('vocab size', tokenizer.vocab_size)

model_hug = hug.BartForConditionalGeneration.from_pretrained(
    model_name).to(settings.DEVICE)

while (not text.isdigit()):
    # text = input("Enter your value: ")
    text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'

    if use_hug:
        tgt_ids = torch.LongTensor([tokenizer.bos_token_id]).unsqueeze(0).to(
            settings.DEVICE)
        input_ids = tokenizer(text, return_tensors='pt').input_ids.to(
            settings.DEVICE)
        if use_encode:
            encoding = model_hug.forward(input_ids, decoder_input_ids=tgt_ids)
            print('encoding', encoding.logits)
        if use_generation:
            generated_ids = model_hug.generate(input_ids)
            print('generated_ids', generated_ids)
            print('hug decoded',
                  tokenizer.batch_decode(generated_ids,
                                         skip_special_tokens=True))

    if use_mgz:
        if use_encode:
            logits = forward_controller(model=model_mgz, text=text,
                                        tokenizer=tokenizer)
            print('encoding', logits)
        if use_generation:
            model_mgz.eval()
            response = generate_controller(model=model_mgz, text=text,
                                           tokenizer=tokenizer)
            print(response)
            print('generated_ids', response)
            print('mgz decoded',
                  tokenizer.batch_decode(response, skip_special_tokens=True))
    exit(3)
    # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    # # to_cuda(model)
