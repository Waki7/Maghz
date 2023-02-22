# from transformers import AutoModelForSequenceClassification
#
# model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased",
#                                                            problem_type="multi_label_classification",
#                                                            num_labels=len(labels),
#                                                            id2label=id2label,
#                                                            label2id=label2id)

from transformers import AutoTokenizer, AutoModel
from transformers.models.bert.tokenization_bert_fast import BertTokenizerFast
from transformers.tokenization_utils_base import BatchEncoding, PreTrainedTokenizerBase

tokenizer: PreTrainedTokenizerBase
tokenizer: BertTokenizerFast = AutoTokenizer.from_pretrained("bert-base-uncased")
model = AutoModel.from_pretrained("bert-base-uncased")
input_text = "Hello world!"
# input_text = ["Hello world!", "f"]
print(tokenizer.tokenize(input_text))
inputs: BatchEncoding = tokenizer(input_text, return_tensors="pt")
print(list(inputs.keys()))
print(list(inputs.values()))
input_ids = inputs['input_ids']
words = tokenizer.batch_decode(input_ids)
print('words', words)
outputs = model(**inputs)

param_size = 0
for param in model.parameters():
    param_size += param.nelement() * param.element_size()
buffer_size = 0
for buffer in model.buffers():
    buffer_size += buffer.nelement() * buffer.element_size()

size_all_mb = (param_size + buffer_size) / 1024**2
print('model size: {:.3f}MB'.format(size_all_mb))
print(model)
# summary(model.cuda(), (INPUT_SHAPE))
# print(model.state_dict())