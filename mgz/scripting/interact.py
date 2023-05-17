from transformers import BartTokenizer
from transformers.modeling_utils import ModelOutput
from mgz.settings import *
from mgz.models.nlp.bart import BartModel, BartForConditionalGeneration
from mgz.models.nlp.tokenizing import Tokenizer, tokenize, TokenStrings

val = 'startval'
tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
from transformers import AutoTokenizer
model: BartForConditionalGeneration = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
while (not val.isdigit()):
    # val = input("Enter your value: ")
    val = 'hello my name is'
    encoded_ids = tokenizer.__call__(val, return_tensors='pt')
    print(encoded_ids)
    model.generation_from_scratch(encoded_ids, tokenizer.pad_token_id, tokenizer.bos_token_id)

    from transformers import BartTokenizer
    # model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
    # # to_cuda(model)
    # text = "Replace me by any text you'd like."
    # output: ModelOutput = model(**encoded_input)
    # # to_cpu(output)
    # print(output)