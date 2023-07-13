import torch
from transformers import BartTokenizer

import settings
from mgz.models.nlp.bart import BartForConditionalGeneration
from mgz.run_ops.run_ops import generate_controller, forward_controller
from mgz.typing import *
import mgz.models.nlp.bart_orig as hug
from transformers import GenerationConfig, BartConfig

with torch.no_grad():

    use_mgz = True
    use_hug = True
    use_generation = True
    use_encode = False
    torch.manual_seed(5)
    text = 'startval'
    # model_name = 'facebook/bart-large'
    # model_name = 'facebook/bart-large-xsum'  # has a bug where it doesn't have 'mask' in its embedding table, or something like that
    # model_name = 'facebook/bart-base'
    model_name = 'allenai/bart-large-multi_lexsum-long-short'
    model_mgz: BartForConditionalGeneration
    model_mgz, tokenizer = BartForConditionalGeneration.from_pretrained(
        model_name, model_name)
    model_mgz.to(settings.DEVICE)
    model_hug = hug.BartForConditionalGeneration.from_pretrained(
        model_name).to(settings.DEVICE)

    while (not text.isdigit()):
        # text = input("Enter your value: ")
        # text = 'The tower is 324 metres (1,063 ft) tall, about the same height as an 81-storey building, and the tallest structure in Paris. Its base is square, measuring 125 metres (410 ft) on each side. During its construction, the Eiffel Tower surpassed the Washington Monument to become the tallest man-made structure in the world, a title it held for 41 years until the Chrysler Building in New York City was finished in 1930. It was the first structure to reach a height of 300 metres. Due to the addition of a broadcasting aerial at the top of the tower in 1957, it is now taller than the Chrysler Building by 5.2 metres (17 ft). Excluding transmitters, the Eiffel Tower is the second tallest free-standing structure in France after the Millau Viaduct.'
        # text = "This case was brought in 2004 by a female former AT&T employee against AT&T Corp. in the U.S. District Court for the Western District of Missouri.  The plaintiff alleged that AT&T, specifically the company's health insurance policy, discriminated against women, and she sought declaratory and injunctive relief, as well as damages.  The Court originally denied the plaintiff's motion for class certification, but later reversed its denial and granted summary judgment to plaintiff, certifying a class to determine compensation.  However, the Court of Appeals referred the District Court Judge to a relevant case which rejected a challenge to a similar program, thereby forcing the Court to vacate its prior ruling and issue judgment in favor of defendants on October 22, 2007."
        text = "On January 23, 2004, Plaintiff filed an amended complaint under Title VII of the Civil Rights Act of 1964 and the Pregnancy Discrimination Act, 42 U.S.C. § 2000e et seq. and 2000e(k), against AT&T Corporation in the United States District Court for the Western District of Missouri.  The plaintiff, represented by private counsel, was a former AT&T employee and asked the Court for declaratory and injunctive relief, as well as damages, alleging that AT&T's health insurance policy discriminated against women.  ", 'On April 25, 2016, three individuals that were blind and enrolled in the Barbri bar exam preparation course filed this putative class action lawsuit in the U.S. District Court for the Northern District of Texas. The plaintiffs brought this suit against BarBri Inc., aka Barbri Bar Review, a company that sells and provides products for bar exam preparation. The plaintiffs alleged that Barbri violated the American with Disabilities Act (ADA) (42 U.S.C. §§ 12111 et seq.) and the Texas Human Resource Code §§ 121'
        text = [text, text]
        batch_size = len(text)
        if use_hug:
            tgt_ids = torch.LongTensor(
                [tokenizer.bos_token_id]).unsqueeze(0).to(
                settings.DEVICE).repeat(batch_size, 1)
            input_ids = tokenizer(text, return_tensors='pt').input_ids.to(
                settings.DEVICE)
            if use_encode:
                encoding = model_hug.forward(input_ids, decoder_input_ids=tgt_ids)
                print('encoding', encoding.logits)
            if use_generation:
                generated_ids = model_hug.generate(input_ids)
                # print('generated_ids', generated_ids)
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
                # print('generated_ids', response)
                print('mgz decoded',
                      tokenizer.batch_decode(response, skip_special_tokens=True))
        exit(3)
        # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
        # # to_cuda(model)
