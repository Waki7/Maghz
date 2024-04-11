import email
import os

import transformers as hug
from nltk.tokenize import sent_tokenize

import mgz.settings as settings
from mgz.typing import *

email = """Message-ID: <28574048.1075852650572.JavaMail.evans@thyme>
Date: Fri, 20 Jul 2001 10:59:25 -0700 (PDT)
From: kevinscott@onlinemailbox.net
To: skean@enron.com
Subject: Moving foward at a good clip
Cc: jeff.skilling@enron.com
Mime-Version: 1.0
Content-Type: text/plain; charset=ANSI_X3.4-1968
Content-Transfer-Encoding: 7bit
Bcc: jeff.skilling@enron.com
X-From: Kevin Scott <kevinscott@onlinemailbox.net>@ENRON <IMCEANOTES-Kevin+20Scott+20+3Ckevinscott+40onlinemailbox+2Enet+3E+40ENRON@ENRON.com>
X-To: Steve Kean <skean@enron.com>
X-cc: Skilling, Jeff </O=ENRON/OU=NA/CN=RECIPIENTS/CN=JSKILLIN>
X-bcc: 
X-Folder: \JSKILLIN (Non-Privileged)\Deleted Items
X-Origin: Skilling-J
X-FileName: JSKILLIN (Non-Privileged).pst


Steve

Good news.  As you indicated would happen, Kalen Pieper called me mid-week.  We had a very good conversation about EES and Dave Delainey's leadership.  She explored my views about doing business with government.  Shortly thereafter, Dave's office called to invite me to Houston on Thursday July 26.  

Kay Chapman explained that Dave's schedule would keep him out of pocket until August 16.  In order to move forward, Janet Dietrich will be meeting with me when I go to Houston next Thursday.  (I must confess that after all the great things I have heard about the man, I do look forward to meeting Dave himself.)

Is there a time next week that I can speak with you by phone to fine-tune my thinking / preparation for Thursday's meeting with Janet?

Thank you for your all of your help.  I am pleased and appreciative that things are moving forward at a good clip.  

Kevin
___________________________________
Contact Information
E-mail
kevinscott@onlinemailbox.net 
Phone
(213) 926-2626
Fax
(707) 516-0019
Traditional Mail
PO Box 21074 ?Los Angeles, CA 90021
___________________________________

"""

B = NewType("Batch", int)
SrcLen = NewType("SrcLen", int)
OutLen = NewType("OutLen", int)

Shape = TypeVar("Shape")
#
# data_dir = '../datasets/enron_export/'
# data: Dict[str, str] = {}
# for root, directories, filenames in os.walk(data_dir):
#     for filename in filenames:
#         if filename.endswith(".txt"):
#             data[filename] = email.message_from_string(
#                 open(os.path.join(root, filename)).read()).as_string()

# chunked: Dict[str, List[str]] = {}
# for key, val in data.items():
#     sentences: List[str] = sent_tokenize(val)
#     # if wanted to combine some sentences
#     chunked[key] = sentences


class Prompts:
    def __init__(self, tokenizer: hug.PreTrainedTokenizer, prompts: List[str],
                 expected_answers: Optional[List[str]]):
        self.tokenizer = tokenizer
        self.prompts = prompts
        self.expected_answers = expected_answers

    def tokenize(self):
        self.tokenize.encode_plus(self.prompts)


def predict_with_model(model: hug.PreTrainedModel,
                       tokenizer: hug.PreTrainedTokenizer, text: List[str]):
    tokenizer_output = tokenizer.batch_encode_plus(text, return_tensors="pt",
                                                   max_length=200)
    src_ids = tokenizer_output.input_ids.to(settings.DEVICE)
    src_mask = tokenizer_output.attention_mask.to(settings.DEVICE)
    output = model.forward(input_ids=src_ids, attention_mask=src_mask,
                           return_dict=True, output_hidden_states=True)
    hidden_states: FloatTensorT['B,SrcSeqLen,EmbedLen'] = output.hidden_states[
                                                              -1][:, -1, :]
    return hidden_states


# node = model_index.ModelDatabase.get_led_model(
#     "allenai/primera-multi_lexsum-source-tiny")
sum_model = hug.LEDForConditionalGeneration.from_pretrained(
    "allenai/primera-multi_lexsum-source-tiny").to(settings.DEVICE)
sum_tokenizer = hug.LEDTokenizerFast.from_pretrained(
    "allenai/primera-multi_lexsum-source-tiny")

encoded = sum_tokenizer.batch_encode_plus([email],
                                          return_tensors="pt")
output = sum_model.generate(encoded.input_ids.to(settings.DEVICE), )
print(sum_tokenizer.batch_decode(output, skip_special_tokens=True))
exit(4)
model = hug.MistralForCausalLM.from_pretrained(
    "openaccess-ai-collective/tiny-mistral")
tokenizer = hug.LlamaTokenizer.from_pretrained(
    "openaccess-ai-collective/tiny-mistral")

print(tokenizer.encode("hello [mask]"))
print(tokenizer.encode("hello"))
print(tokenizer.pad_token)
print(tokenizer.mask_token)

tokenizer.pad_token_id = tokenizer.eos_token_id
encodings: TensorType('B,SrcLen') = tokenizer.apply_chat_template([
    {"role": "user", "content": "Hello, world! What is your name"}
], return_tensors="pt", max_length=50, return_length=True)
print(encodings.shape)

generated: TensorType('B,OutLen') = model.generate(encodings,
                                                   max_length=50 + 50, )

print(tokenizer.decode(generated[0], skip_special_tokens=True))

# model_inference(model, torch.Tensor([1]))
