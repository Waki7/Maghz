from __future__ import annotations

from enum import Enum

import transformers as hug

import settings
from mgz.model_running.run_ops import generate_controller, forward_controller
from mgz.typing import *
from mgz.version_control.model_node import lookup_model


class Model(Enum):
    BART = 'BART'
    LED = 'LED'


def model_selectors(model_typ: Model, use_mgz: bool = False,
                    use_hug: bool = False):
    model = None
    tokenizer = None
    if model_typ == Model.LED:
        model_name = 'allenai/led-base-16384-multi_lexsum-source-long'
        if use_hug:
            tokenizer = hug.LEDTokenizerFast.from_pretrained(
                model_name)
            model = hug.LEDForConditionalGeneration.from_pretrained(
                model_name)
        if use_mgz:
            from mgz.models.nlp.led import LEDForConditionalGeneration
            node = lookup_model(LEDForConditionalGeneration,
                model_name, model_name)
            model, tokenizer = node.model, node.tokenizer
    if model_typ == Model.BART:
        model_name = 'allenai/bart-large-multi_lexsum-long-short'
        if use_hug:
            tokenizer = hug.BartTokenizerFast.from_pretrained(model_name)
            model = hug.BartForConditionalGeneration.from_pretrained(
                model_name)
        if use_mgz:
            from mgz.models.nlp.bart import BartForConditionalGeneration
            node = lookup_model(BartForConditionalGeneration,
                model_name, model_name)
            model, tokenizer = node.model, node.tokenizer
    return model.to(settings.DEVICE), tokenizer


with torch.no_grad():
    use_mgz = False
    use_hug = False

    use_mgz = True
    use_hug = True
    use_generation = True  # True False
    use_encode = False  # True False
    use_model = Model.LED

    torch.manual_seed(5)
    text = 'startval'

    # BART
    # model_name = 'facebook/bart-large'
    # model_name = 'facebook/bart-large-xsum'  # has a bug where it doesn't have 'mask' in its embedding table, or something like that
    # model_name = 'facebook/bart-base'
    model_name = 'allenai/bart-large-multi_lexsum-long-short'
    model_name = 'allenai/led-base-16384-multi_lexsum-source-long'

    while (not text.isdigit()):
        # text = input("Enter your value: ")
        # text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'
        # text = "This case was brought in 2004 by a female former AT&T employee against AT&T Corp. in the U.S. District Court for the Western District of Missouri.  The plaintiff alleged that AT&T, specifically the company's health insurance policy, discriminated against women, and she sought declaratory and injunctive relief, as well as damages.  The Court originally denied the plaintiff's motion for class certification, but later reversed its denial and granted summary judgment to plaintiff, certifying a class to determine compensation.  However, the Court of Appeals referred the District Court Judge to a relevant case which rejected a challenge to a similar program, thereby forcing the Court to vacate its prior ruling and issue judgment in favor of defendants on October 22, 2007."
        text = "On January 23, 2004, Plaintiff filed an amended complaint under Title VII of the Civil Rights Act of 1964 and the Pregnancy Discrimination Act, 42 U.S.C. § 2000e et seq. and 2000e(k), against AT&T Corporation in the United States District Court for the Western District of Missouri.  The plaintiff, represented by private counsel, was a former AT&T employee and asked the Court for declaratory and injunctive relief, as well as damages, alleging that AT&T's health insurance policy discriminated against women.  ", 'On April 25, 2016, three individuals that were blind and enrolled in the Barbri bar exam preparation course filed this putative class action lawsuit in the U.S. District Court for the Northern District of Texas. The plaintiffs brought this suit against BarBri Inc., aka Barbri Bar Review, a company that sells and provides products for bar exam preparation. The plaintiffs alleged that Barbri violated the American with Disabilities Act (ADA) (42 U.S.C. §§ 12111 et seq.) and the Texas Human Resource Code §§ 121'
        text = '. '.join(
            [text[0], text[0], text[0], text[0], text[0], text[0], text[0],
             text[0], text[0], text[0]])
        text = [text, text[0:-10]]
        batch_size = len(text)
        if use_hug:
            model_hug, tokenizer = model_selectors(use_model, use_hug=True)
            tgt_ids = torch.LongTensor(
                [tokenizer.sep_token_id]).unsqueeze(0).to(
                settings.DEVICE).repeat(batch_size, 1)
            print(model_hug.config.max_length)
            input_ids = tokenizer(text, return_tensors='pt',
                                  padding=True).input_ids.to(
                settings.DEVICE)
            if use_encode:
                encoding = model_hug.forward(input_ids,
                                             decoder_input_ids=tgt_ids)
                print(
                    'encoding with shape {} \n {}'.format(encoding.logits.shape,
                                                          encoding.logits))
            if use_generation:
                generated_ids = model_hug.generate(input_ids)
                # print('generated_ids', generated_ids)
                summary: List[str] = tokenizer.batch_decode(generated_ids,
                                                            skip_special_tokens=True)
                print('hug decoded', summary)

        if use_mgz:
            model_mgz, tokenizer = model_selectors(use_model, use_mgz=True)
            if use_encode:
                logits = forward_controller(model=model_mgz, text=text,
                                            tokenizer=tokenizer)
                print(
                    'encoding with shape {} \n {}'.format(logits.shape, logits))

                # embedding = embedding_controller(model_mgz, text, tokenizer)
            if use_generation:
                model_mgz.eval()
                response = generate_controller(model=model_mgz, text=text,
                                               tokenizer=tokenizer)
                # print('generated_ids', response)
                summary: List[str] = tokenizer.batch_decode(response,
                                                            skip_special_tokens=True)
                print('mgz decoded', summary)
        exit(3)
        # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
        # # to_cuda(model)
