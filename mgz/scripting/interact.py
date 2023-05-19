import torch
from nptyping import NDArray, Shape, Int32
from transformers import BartTokenizer

import settings
from mgz.models.nlp.bart import BartForConditionalGeneration
from mgz.run_ops.run_ops import generate
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

val = 'startval'
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
print('bos', tokenizer.bos_token_id)
print('pad', tokenizer.pad_token_id)
print('eos', tokenizer.eos_token_id)
model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained(
    "facebook/bart-base").to(settings.DEVICE)
from transformers import BartConfig

# model: BartForConditionalGeneration = BartForConditionalGeneration(BartConfig()).to(settings.DEVICE)

while (not val.isdigit()):
    # val = input("Enter your value: ")
    val = 'the end of the story'
    response = generate(val, tokenizer, model)
    print(response)
    print(tokenizer.decode(response[0], skip_special_tokens=True))
    exit(3)
    # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    # # to_cuda(model)
    # text = "Replace me by any text you'd like."
    # output: ModelOutput = model(**encoded_input)
    # # to_cpu(output)
    # print(output)
