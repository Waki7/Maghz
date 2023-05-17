from transformers import BartTokenizer
from transformers.modeling_utils import ModelOutput
from mgz.settings import *
from mgz.models.nlp.bart import BartModel

val = 'startval'
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
tokens = tokenizer.tokenize(val)
model: BartModel = BartModel.from_pretrained("facebook/bart-base")
while (not val.isdigit()):
    val = input("Enter your value: ")
    from transformers import BartTokenizer
    # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    # # to_cuda(model)
    # text = "Replace me by any text you'd like."
    # encoded_input = tokenizer(text, return_tensors='pt')
    # output: ModelOutput = model(**encoded_input)
    # # to_cpu(output)
    # print(output)