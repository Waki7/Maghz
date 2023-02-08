from transformers import BertTokenizer, BertModel, BertForQuestionAnswering
from transformers.modeling_utils import ModelOutput
from mgz.settings import *
tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
# model: BertModel = BertModel.from_pretrained("bert-base-cased")
model: BertModel = BertForQuestionAnswering.from_pretrained("bert-base-cased")
# to_cuda(model)
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt')
output: ModelOutput = model(**encoded_input)
# to_cpu(output)
print(output)