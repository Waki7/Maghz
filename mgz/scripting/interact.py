import torch
from transformers import BartTokenizer

import settings
from mgz.models.nlp.bart import BartForConditionalGeneration
from mgz.run_ops.run_ops import generate_controller, forward_controller
from mgz.typing import *

#
# def test(tensor: LongTensorT['3,3,4']) -> LongTensorT['3,3,4']:
#     print(tensor.shape)
#     return torch.Tensor([1, 2, 3, 4])
# def testnd(tensor: NDArray[Shape["4"], Int32])-> NDArray[Shape["4"], Int32]:
#     print(tensor.shape)
#     return np.array([1, 2, 3, 4])
#
#
# testnd( np.array([1, 2, 3, 4]))
# test(torch.Tensor([1, 2, 3, 4]))

text = 'startval'
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
print('bos', tokenizer.bos_token_id)
print('pad', tokenizer.pad_token_id)
print('eos', tokenizer.eos_token_id)
model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base").to(settings.DEVICE)

import transformers as hug

model2 = hug.BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base").to(settings.DEVICE)
# model: BartForConditionalGeneration = BartForConditionalGeneration(BartConfig()).to(settings.DEVICE)

while (not text.isdigit()):
    # val = input("Enter your value: ")
    text = 'what is the end of the <mask>'
    # print(model)
    # print(model2)
    if True:
        transformer_output = model2.forward(
            tokenizer(text, return_tensors='pt').input_ids.to(settings.DEVICE))
        print(transformer_output.logits.shape)
        print(transformer_output.logits)
        # generated_ids = model2.generate(
        #     tokenizer(text, return_tensors='pt').input_ids.to(settings.DEVICE))
        # print(generated_ids)
        # print(tokenizer.decode(generated_ids[0], skip_special_tokens=True))
    if True:
        response = forward_controller(model=model, text=text, tokenizer=tokenizer)
        print(response.shape)
        print(response)
        # response = generate_controller(model=model, text=text, tokenizer=tokenizer)

    exit(3)
    # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    # # to_cuda(model)
    # text = "Replace me by any text you'd like."
    # output: ModelOutput = model(**encoded_input)
    # # to_cpu(output)
    # print(output)
