from typing import *

import torch
import transformers as hug

B = NewType("Batch", int)
SrcLen = NewType("SrcLen", int)
OutLen = NewType("OutLen", int)

Shape = TypeVar("Shape")

class DataPromptsDateQuestions:
  def __init__(self):
    self.length = 100
    self.prompts = ["Hello, world! What is your name?"]

  def __getitem__(self, item: int):
    return "Hello, world! What is your name?"




def add_token(tokenizer: hug.PreTrainedTokenizer, model: hug.PreTrainedModel, tokens):
  tokenizer.add_tokens(tokens, special_tokens=False)
  model.resize_token_embeddings(len(tokenizer))


def TensorType(shapes: Generic[Shape]):
    return NewType(shapes, torch.Tensor)


def model_inference(model: hug.PreTrainedModel, x: TensorType('B,Optional[1]')):
    print(model)
    print(x)


model = hug.MistralForCausalLM.from_pretrained("openaccess-ai-collective/tiny-mistral")
tokenizer = hug.LlamaTokenizer.from_pretrained("openaccess-ai-collective/tiny-mistral")
print(tokenizer.encode("hello [mask]"))
print(tokenizer.encode("hello"))
print(tokenizer.pad_token)
print(tokenizer.mask_token)
exit(3)


tokenizer.pad_token_id = tokenizer.eos_token_id
encodings: TensorType('B,SrcLen') = tokenizer.apply_chat_template([
  {"role": "user", "content": "Hello, world! What is your name"}
], return_tensors="pt", max_length=50, return_length=True)
print(encodings.shape)

generated: TensorType('B,OutLen') = model.generate(encodings, max_length=50+50,  )

print(tokenizer.decode(generated[0], skip_special_tokens=True))






# model_inference(model, torch.Tensor([1]))
